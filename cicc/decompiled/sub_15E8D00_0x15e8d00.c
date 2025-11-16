// Function: sub_15E8D00
// Address: 0x15e8d00
//
_QWORD *__fastcall sub_15E8D00(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        __int64 *a4,
        int a5,
        int a6,
        unsigned int a7,
        char *a8,
        __int64 a9,
        char *a10,
        __int64 a11,
        char *a12,
        __int64 a13,
        char *a14,
        __int64 a15,
        __int64 a16)
{
  __int64 *v20; // rdi
  _QWORD *v21; // r12
  __int64 v24; // [rsp+18h] [rbp-58h]
  __int64 v25[10]; // [rsp+20h] [rbp-50h] BYREF

  v20 = *(__int64 **)(*(_QWORD *)(a1[1] + 56) + 40LL);
  v25[0] = *a4;
  v24 = sub_15E26F0(v20, 78, v25, 1);
  sub_15E8A10((__int64)v25, (__int64)a1, a2, a3, (__int64)a4, a7, a8, a9, a10, a11, a12, a13, a14, a15);
  v21 = sub_15E6C70(v24, a5, a6, v25[0], (v25[1] - v25[0]) >> 3, a1, a16);
  if ( v25[0] )
    j_j___libc_free_0(v25[0], v25[2] - v25[0]);
  return v21;
}
