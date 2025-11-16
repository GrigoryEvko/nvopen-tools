// Function: sub_15E8C20
// Address: 0x15e8c20
//
_QWORD *__fastcall sub_15E8C20(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        __int64 *a4,
        unsigned int a5,
        int a6,
        char *a7,
        __int64 a8,
        char *a9,
        __int64 a10,
        char *a11,
        __int64 a12,
        char *a13,
        __int64 a14)
{
  __int64 *v17; // rdi
  __int64 v18; // r13
  _QWORD *v19; // r12
  __int64 v23; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v24[10]; // [rsp+20h] [rbp-50h] BYREF

  v17 = *(__int64 **)(*(_QWORD *)(a1[1] + 56) + 40LL);
  v23 = *a4;
  v18 = sub_15E26F0(v17, 78, &v23, 1);
  sub_15E8A10((__int64)v24, (__int64)a1, a2, a3, (__int64)a4, a5, a7, a8, a9, a10, a11, a12, a13, a14);
  v19 = sub_15E6DE0(v18, v24[0], (__int64)(v24[1] - v24[0]) >> 3, a1, a6, 0, 0, 0);
  if ( v24[0] )
    j_j___libc_free_0(v24[0], v24[2] - v24[0]);
  return v19;
}
