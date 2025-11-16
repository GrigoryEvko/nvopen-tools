// Function: sub_270A460
// Address: 0x270a460
//
bool __fastcall sub_270A460(__int64 a1)
{
  return sub_BA8B30(a1, (__int64)"llvm.objc.retain", 0x10u)
      || sub_BA8B30(a1, (__int64)"llvm.objc.release", 0x11u)
      || sub_BA8B30(a1, (__int64)"llvm.objc.autorelease", 0x15u)
      || sub_BA8B30(a1, (__int64)"llvm.objc.retainAutoreleasedReturnValue", 0x27u)
      || sub_BA8B30(a1, (__int64)"llvm.objc.unsafeClaimAutoreleasedReturnValue", 0x2Cu)
      || sub_BA8B30(a1, (__int64)"llvm.objc.retainBlock", 0x15u)
      || sub_BA8B30(a1, (__int64)"llvm.objc.autoreleaseReturnValue", 0x20u)
      || sub_BA8B30(a1, (__int64)"llvm.objc.autoreleasePoolPush", 0x1Du)
      || sub_BA8B30(a1, (__int64)"llvm.objc.loadWeakRetained", 0x1Au)
      || sub_BA8B30(a1, (__int64)"llvm.objc.loadWeak", 0x12u)
      || sub_BA8B30(a1, (__int64)"llvm.objc.destroyWeak", 0x15u)
      || sub_BA8B30(a1, (__int64)"llvm.objc.storeWeak", 0x13u)
      || sub_BA8B30(a1, (__int64)"llvm.objc.initWeak", 0x12u)
      || sub_BA8B30(a1, (__int64)"llvm.objc.moveWeak", 0x12u)
      || sub_BA8B30(a1, (__int64)"llvm.objc.copyWeak", 0x12u)
      || sub_BA8B30(a1, (__int64)"llvm.objc.retainedObject", 0x18u)
      || sub_BA8B30(a1, (__int64)"llvm.objc.unretainedObject", 0x1Au)
      || sub_BA8B30(a1, (__int64)"llvm.objc.unretainedPointer", 0x1Bu)
      || sub_BA8B30(a1, (__int64)"llvm.objc.clang.arc.noop.use", 0x1Cu)
      || sub_BA8B30(a1, (__int64)"llvm.objc.clang.arc.use", 0x17u) != 0;
}
