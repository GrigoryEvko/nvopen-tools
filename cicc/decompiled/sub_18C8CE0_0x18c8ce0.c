// Function: sub_18C8CE0
// Address: 0x18c8ce0
//
bool __fastcall sub_18C8CE0(__int64 a1)
{
  return sub_1632000(a1, (__int64)"objc_retain", 11)
      || sub_1632000(a1, (__int64)"objc_release", 12)
      || sub_1632000(a1, (__int64)"objc_autorelease", 16)
      || sub_1632000(a1, (__int64)"objc_retainAutoreleasedReturnValue", 34)
      || sub_1632000(a1, (__int64)"objc_unsafeClaimAutoreleasedReturnValue", 39)
      || sub_1632000(a1, (__int64)"objc_retainBlock", 16)
      || sub_1632000(a1, (__int64)"objc_autoreleaseReturnValue", 27)
      || sub_1632000(a1, (__int64)"objc_autoreleasePoolPush", 24)
      || sub_1632000(a1, (__int64)"objc_loadWeakRetained", 21)
      || sub_1632000(a1, (__int64)"objc_loadWeak", 13)
      || sub_1632000(a1, (__int64)"objc_destroyWeak", 16)
      || sub_1632000(a1, (__int64)"objc_storeWeak", 14)
      || sub_1632000(a1, (__int64)"objc_initWeak", 13)
      || sub_1632000(a1, (__int64)"objc_moveWeak", 13)
      || sub_1632000(a1, (__int64)"objc_copyWeak", 13)
      || sub_1632000(a1, (__int64)"objc_retainedObject", 19)
      || sub_1632000(a1, (__int64)"objc_unretainedObject", 21)
      || sub_1632000(a1, (__int64)"objc_unretainedPointer", 22)
      || sub_1632000(a1, (__int64)"clang.arc.use", 13) != 0;
}
