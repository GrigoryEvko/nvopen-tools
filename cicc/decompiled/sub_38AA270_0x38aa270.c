// Function: sub_38AA270
// Address: 0x38aa270
//
__int64 __fastcall sub_38AA270(
        __int64 a1,
        _DWORD *a2,
        __int64 *a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  _BYTE *v13; // rsi
  __int64 v14; // rdx
  double v15; // xmm4_8
  double v16; // xmm5_8
  unsigned int v17; // r12d
  const void *v19[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v20[6]; // [rsp+10h] [rbp-30h] BYREF

  v13 = *(_BYTE **)(a1 + 72);
  v14 = *(_QWORD *)(a1 + 80);
  v19[0] = v20;
  sub_3887850((__int64 *)v19, v13, (__int64)&v13[v14]);
  *a2 = sub_1632050(*(__int64 ***)(a1 + 176), v19[0], (size_t)v19[1]);
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  v17 = sub_38AA210(a1, a3, a4, a5, a6, a7, v15, v16, a10, a11);
  if ( v19[0] != v20 )
    j_j___libc_free_0((unsigned __int64)v19[0]);
  return v17;
}
