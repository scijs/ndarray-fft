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
  
  /*
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
  */
  
  test_random([10, 8])
  
  t.end()
})