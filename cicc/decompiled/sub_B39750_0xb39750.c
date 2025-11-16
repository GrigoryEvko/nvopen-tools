// Function: sub_B39750
// Address: 0xb39750
//
__int64 __fastcall sub_B39750(
        __int64 a1,
        char *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        __int64 a8,
        char a9,
        _QWORD *a10,
        __int64 a11,
        char a12)
{
  char *v14; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v15; // [rsp+8h] [rbp-B8h]
  char v16[176]; // [rsp+10h] [rbp-B0h] BYREF

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  if ( !a12
    || (v15 = 0x1000000000LL,
        v14 = v16,
        sub_B32D80((__int64)&v14, a10, a11),
        sub_B38780((__m128i **)a1, "deopt", (__int64)&v14),
        v14 == v16) )
  {
    if ( !a9 )
      goto LABEL_3;
  }
  else
  {
    _libc_free(v14, "deopt");
    if ( !a9 )
      goto LABEL_3;
  }
  v15 = 0x1000000000LL;
  v14 = v16;
  sub_B32D80((__int64)&v14, a7, a8);
  sub_B38E70((__m128i **)a1, "gc-transition", (__int64)&v14);
  if ( v14 == v16 )
  {
LABEL_3:
    if ( !a3 )
      return a1;
    goto LABEL_9;
  }
  _libc_free(v14, "gc-transition");
  if ( !a3 )
    return a1;
LABEL_9:
  v15 = 0x1000000000LL;
  v14 = v16;
  sub_B32F20((__int64)&v14, v16, a2, &a2[8 * a3]);
  sub_B39560((__m128i **)a1, "gc-live", (__int64)&v14);
  if ( v14 != v16 )
    _libc_free(v14, "gc-live");
  return a1;
}
