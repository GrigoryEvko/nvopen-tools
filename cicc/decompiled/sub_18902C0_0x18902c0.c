// Function: sub_18902C0
// Address: 0x18902c0
//
__int64 __fastcall sub_18902C0(
        __int64 a1,
        _QWORD **a2,
        __m128 a3,
        __m128 a4,
        __m128i a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  double v11; // xmm4_8
  double v12; // xmm5_8
  unsigned int v13; // eax
  __int64 v14; // r13
  __int64 v15; // r12
  unsigned int v16; // r14d
  __int64 v17; // rbx
  __int64 v18; // rdi
  _QWORD *v19; // rbx
  _QWORD *v20; // r12
  __int64 v21; // rdi
  __int64 *v22[15]; // [rsp-D8h] [rbp-D8h] BYREF
  _QWORD *v23; // [rsp-60h] [rbp-60h]
  unsigned int v24; // [rsp-50h] [rbp-50h]
  __int64 v25; // [rsp-48h] [rbp-48h]
  __int64 v26; // [rsp-40h] [rbp-40h]
  __int64 v27; // [rsp-38h] [rbp-38h]

  if ( *(_BYTE *)(a1 + 153) )
    return sub_188F8B0(a2, a3, a4, a5, a6, a7, a8, a9, a10);
  sub_1875960(v22, a2, *(__int64 **)(a1 + 160), *(__int64 **)(a1 + 168));
  v13 = sub_188C730((__int64 *)v22, a3, a4, a5, a6, v11, v12, a9, a10);
  v14 = v26;
  v15 = v25;
  v16 = v13;
  if ( v26 != v25 )
  {
    do
    {
      v17 = *(_QWORD *)(v15 + 16);
      while ( v17 )
      {
        sub_1876060(*(_QWORD *)(v17 + 24));
        v18 = v17;
        v17 = *(_QWORD *)(v17 + 16);
        j_j___libc_free_0(v18, 40);
      }
      v15 += 80;
    }
    while ( v14 != v15 );
    v15 = v25;
  }
  if ( v15 )
    j_j___libc_free_0(v15, v27 - v15);
  if ( v24 )
  {
    v19 = v23;
    v20 = &v23[5 * v24];
    do
    {
      if ( *v19 != -8 && *v19 != -4 )
      {
        v21 = v19[1];
        if ( v21 )
          j_j___libc_free_0(v21, v19[3] - v21);
      }
      v19 += 5;
    }
    while ( v20 != v19 );
  }
  j___libc_free_0(v23);
  return v16;
}
