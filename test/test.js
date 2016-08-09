"use strict"

var ndfft = require("../fft.js")
var ndarray = require("ndarray")
var ops = require("ndarray-ops")
var zeros = require("zeros")
var almostEqual = require("almost-equal")
var arrayAlmostEqual = require("array-almost-equal")

var EPSILON = almostEqual.FLT_EPSILON

require("tape")("ndarray-fft", function(t) {
  
  function test_spike(n) {
    var i, j
    function eq(a, b, str) {
      t.assert(almostEqual(a, b, EPSILON, EPSILON), str + "/n=" + n + ", i=" + i + " - got: " + a + ", expect: " + b)
    }
    var x = ndarray(new Float64Array(n))
    var xr = ndarray(new Float64Array(n))
    var y = ndarray(new Float64Array(n))
    var yr = ndarray(new Float64Array(n))
    for(i=0; i<n; ++i) {
      for(j=0; j<n; ++j) {
        x.set(j, 0.0)
        y.set(j, 0.0)
      }
      x.set(i, 1.0)
      ndfft(1, x, y)
      for(j=0; j<n; ++j) {
        var a = 2.0 * Math.PI * i * j / n
        xr.set(j, Math.cos(a))
        yr.set(j, Math.sin(a))
      }
      t.assert(arrayAlmostEqual(x.data, xr.data), "1D Forward real part (n="+n+").")
      t.assert(arrayAlmostEqual(y.data, yr.data), "1D Forward imaginary part (n="+n+").")
      ndfft(-1, x, y)
      for(j=0; j<n; ++j) {
        if(j === i) {
          xr.set(j, 1.0)
        } else {
          xr.set(j, 0.0)
        }
        yr.set(j, 0.0)
      }
      t.assert(arrayAlmostEqual(x.data, xr.data), "1D Inverse real part (n="+n+").")
      t.assert(arrayAlmostEqual(y.data, yr.data), "1D Inverse imaginary part (n="+n+").")
    }
  }
  
  function test_spike2(n, m) {
    var i1, i2, j1, j2
    function eq(ar, ai, br, bi, str) {
      t.assert(almostEqual(ar, br, EPSILON, EPSILON) &&
               almostEqual(ai, bi, EPSILON, EPSILON),
               str + "/n=(" + n + "," + m + "), i=(" + i1 + "," + i2 + "), j=(" + j1 + "," + j2 + "), - got: " + ar + " + " + ai + "i, expect: " + br + " + " + bi + "i")
    }
    var x = zeros([n, m])
    var xr = zeros([n, m])
    var y = zeros([n, m])
    var yr = zeros([n, m])
    for(i1=0; i1<n; ++i1) {
      for(i2=0; i2<m; ++i2) {
        ops.assigns(x, 0.0)
        ops.assigns(y, 0.0)
        x.set(i1, i2, 1.0)
        
        ndfft(1, x, y)
        for(j1=0; j1<n; ++j1) {
          var a = 2.0 * Math.PI * i1 * j1 / n
          for(j2=0; j2<m; ++j2) {
            var b = 2.0 * Math.PI * i2 * j2 / m
            xr.set(j1,j2, Math.cos(a) * Math.cos(b) - Math.sin(a) * Math.sin(b))
            yr.set(j1,j2, Math.sin(a) * Math.cos(b) + Math.cos(a) * Math.sin(b))
          }
        }
        t.assert(arrayAlmostEqual(x.data, xr.data), "2D Forward real part (n="+n+",m="+m+").")
        t.assert(arrayAlmostEqual(y.data, yr.data), "2D Forward imaginary part (n="+n+",m="+m+").")

        ndfft(-1, x, y)        
        for(j1=0; j1<n; ++j1) {
          for(j2=0; j2<m; ++j2) {
            if(j1 === i1 && j2 === i2) {
              xr.set(j1,j2, 1.0)
            } else {
              xr.set(j1,j2, 0.0)
            }
            yr.set(j1,j2, 0.0)
          }
        }
        t.assert(arrayAlmostEqual(x.data, xr.data), "2D Forward real part (n="+n+",m="+m+").")
        t.assert(arrayAlmostEqual(y.data, yr.data), "2D Forward imaginary part (n="+n+",m="+m+").")
      }
    }
  }
  
  
  function test_random(shape) {
    var i
    function eq(a, b) {
      t.assert(almostEqual(a, b, EPSILON, EPSILON),"rnd/n=" + shape + ", i=" + i + " - got: " + a + ", expect: " + b)
    }
    var x = zeros(shape)
      , y = zeros(shape)
      
    ops.random(x)
    ops.random(y)
    
    var xp = zeros(shape)
      , yp = zeros(shape)
    ops.assign(xp, x)
    ops.assign(yp, y)
    
    ndfft(1, x, y)
    ndfft(-1, x, y)
    
    t.assert(arrayAlmostEqual(x.data, xp.data), "Random round-trip real part (n="+shape +").")
    t.assert(arrayAlmostEqual(y.data, yp.data), "Random round-trip imaginary part (n="+shape +").")
  }
  
  test_spike(1)
  test_spike(2)
  test_spike(4)
  test_spike(16)
  test_spike(32)
  test_spike(3)
  test_spike(5)
  test_spike(6)
  test_spike(7)
  test_spike(10)
  test_spike(15)
  test_spike(17)
  
  for(var i=1; i<100; ++i) {
    test_random([i])
  }
  
  test_spike2(2, 2)
  test_spike2(4, 2)
  test_spike2(2, 4)
  test_spike2(4, 4)
  test_spike2(8, 8)
  test_spike2(3, 3)
  test_spike2(5, 5)
  test_spike2(7, 7)
  
  for(var i=1; i<=8; ++i) {
    for(var j=1; j<=8; ++j) {
      test_random([i, j])
    }
  }
  test_random([100, 100])
  test_random([32, 32, 32])
  test_random([15, 15, 15])
  test_random([8, 7, 5, 4])
  
  t.end()
})
