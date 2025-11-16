// Function: sub_BCFD60
// Address: 0xbcfd60
//
__int64 __fastcall sub_BCFD60(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        const void *a4,
        __int64 a5,
        __int64 a6,
        void *a7,
        __int64 a8)
{
  __int64 v9; // [rsp+0h] [rbp-20h] BYREF
  char v10; // [rsp+8h] [rbp-18h]

  sub_BCFB10((__int64)&v9, a1, a2, a3, a4, a5, a7, a8);
  if ( (v10 & 1) != 0 )
    BUG();
  return v9;
}
