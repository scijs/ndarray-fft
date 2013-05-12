"use strict"

var ndfft = require("../fft.js")
var ndarray = require("ndarray")
var ops = require("ndarray-ops")
var almostEqual = require("almost-equal")

var EPSILON = almostEqual.FLT_EPSILON

require("tap").test("ndarray-fft", function(t) {
  
  function test_spike(n) {
    var i, j
    function eq(a, b, str) {
      t.assert(almostEqual(a, b, EPSILON, EPSILON), str + "/n=" + n + ", i=" + i + " - got: " + a + ", expect: " + b)
    }
    var x = ndarray(new Float64Array(n))
    var y = ndarray(new Float64Array(n))
    for(i=0; i<n; ++i) {
      for(j=0; j<n; ++j) {
        x.set(j, 0.0)
        y.set(j, 0.0)
      }
      x.set(i, 1.0)
      ndfft(1, x, y)
      for(j=0; j<n; ++j) {
        var a = 2.0 * Math.PI * i * j / n
        eq(x.get(j), Math.cos(a), "fwd")
        eq(y.get(j), Math.sin(a), "fwd")
      }
      ndfft(-1, x, y)
      for(j=0; j<n; ++j) {
        if(j === i) {
          eq(x.get(j), 1.0, "inv")
        } else {
          eq(x.get(j), 0.0, "inv")
        }
        eq(y.get(j), 0.0, "inv")
      }
    }
  }
  
  function test_spike2(n, m) {
    var i1, i2, j1, j2
    function eq(ar, ai, br, bi, str) {
      t.assert(almostEqual(ar, br, EPSILON, EPSILON) &&
               almostEqual(ai, bi, EPSILON, EPSILON),
               str + "/n=(" + n + "," + m + "), i=(" + i1 + "," + i2 + "), j=(" + j1 + "," + j2 + "), - got: " + ar + " + " + ai + "i, expect: " + br + " + " + bi + "i")
    }
    var x = ndarray.zeros([n, m])
    var y = ndarray.zeros([n, m])
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
            eq( x.get(j1, j2),
                y.get(j1, j2),
                Math.cos(a) * Math.cos(b) - Math.sin(a) * Math.sin(b),
                Math.sin(a) * Math.cos(b) + Math.cos(a) * Math.sin(b),
                "fwd" )
          }
        }
        ndfft(-1, x, y)
        
        for(j1=0; j1<n; ++j1) {
          for(j2=0; j2<m; ++j2) {
            if(j1 === i1 && j2 === i2) {
              eq(x.get(j1, j2), y.get(j1, j2), 1.0, 0.0, "inv")
            } else {
              eq(x.get(j1, j2), y.get(j1, j2), 0.0, 0.0, "inv")
            }
          }
        }
      }
    }
  }
  
  
  function test_random(shape) {
    var i
    function eq(a, b) {
      t.assert(almostEqual(a, b, EPSILON, EPSILON),"rnd/n=" + shape + ", i=" + i + " - got: " + a + ", expect: " + b)
    }
    var x = ndarray.zeros(shape)
      , y = ndarray.zeros(shape)
      
    ops.random(x)
    ops.random(y)
    
    var xp = ops.clone(x)
      , yp = ops.clone(y)
    
    ndfft(1, x, y)
    ndfft(-1, x, y)
    
    for(i=0; i<x.data.length; ++i) {
      eq(x.data[i], xp.data[i])
      eq(y.data[i], yp.data[i])
    }
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
