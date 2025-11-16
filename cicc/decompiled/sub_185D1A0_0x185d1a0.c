// Function: sub_185D1A0
// Address: 0x185d1a0
//
__int64 __fastcall sub_185D1A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 ****v12; // rbx
  __int64 ****i; // r13
  __int64 ***v14; // r12
  double v15; // xmm4_8
  double v16; // xmm5_8
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  __int64 v19; // rbx
  unsigned __int64 v20; // r13
  __int64 v21; // r12
  double v22; // xmm4_8
  double v23; // xmm5_8
  unsigned __int64 v24; // rdi
  __int64 v25; // r15
  __int64 v26; // r12
  __int64 *v27; // r13
  __int64 v28; // r14
  __int64 v29; // rbx
  __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 v32; // rdi
  __int64 result; // rax
  __int64 v34; // rdi
  __int64 *v35; // rbx
  unsigned __int64 v36; // r12
  __int64 v37; // rdi
  __int64 v38; // rdi
  __int64 j; // [rsp+8h] [rbp-58h]
  __int64 v40; // [rsp+10h] [rbp-50h]
  unsigned __int64 v41; // [rsp+18h] [rbp-48h]
  __int64 v42; // [rsp+20h] [rbp-40h]

  v12 = *(__int64 *****)(a1 + 160);
  for ( i = &v12[*(unsigned int *)(a1 + 168)]; i != v12; ++v12 )
  {
    v14 = *v12;
    if ( (*v12)[1] )
    {
      a2 = sub_15A06D0(*v14, a2, a3, a4);
      sub_164D160((__int64)v14, a2, a5, a6, a7, a8, v15, v16, a11, a12);
    }
  }
  v17 = *(_QWORD *)(a1 + 552);
  if ( v17 != *(_QWORD *)(a1 + 544) )
    _libc_free(v17);
  v18 = *(_QWORD *)(a1 + 448);
  if ( v18 != *(_QWORD *)(a1 + 440) )
    _libc_free(v18);
  v19 = *(_QWORD *)(a1 + 160);
  v20 = v19 + 8LL * *(unsigned int *)(a1 + 168);
  if ( v19 != v20 )
  {
    do
    {
      v21 = *(_QWORD *)(v20 - 8);
      v20 -= 8LL;
      if ( v21 )
      {
        sub_15E5530(v21);
        sub_159D9E0(v21);
        sub_164BE60(v21, a5, a6, a7, a8, v22, v23, a11, a12);
        *(_DWORD *)(v21 + 20) = *(_DWORD *)(v21 + 20) & 0xF0000000 | 1;
        sub_1648B90(v21);
      }
    }
    while ( v19 != v20 );
    v20 = *(_QWORD *)(a1 + 160);
  }
  if ( v20 != a1 + 176 )
    _libc_free(v20);
  j___libc_free_0(*(_QWORD *)(a1 + 136));
  v24 = *(_QWORD *)(a1 + 80);
  if ( v24 != a1 + 96 )
    _libc_free(v24);
  v25 = *(_QWORD *)(a1 + 56);
  v26 = *(_QWORD *)(a1 + 16);
  v42 = *(_QWORD *)(a1 + 48);
  v41 = *(_QWORD *)(a1 + 72);
  v40 = *(_QWORD *)(a1 + 32);
  v27 = (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL);
  for ( j = *(_QWORD *)(a1 + 40); v41 > (unsigned __int64)v27; ++v27 )
  {
    v28 = *v27;
    v29 = *v27 + 512;
    do
    {
      v30 = *(_QWORD *)(v28 + 8);
      v28 += 32;
      j___libc_free_0(v30);
    }
    while ( v29 != v28 );
  }
  if ( v41 == j )
  {
    while ( v42 != v26 )
    {
      v38 = *(_QWORD *)(v26 + 8);
      v26 += 32;
      j___libc_free_0(v38);
    }
    result = a1;
    v34 = *(_QWORD *)a1;
    if ( !*(_QWORD *)a1 )
      return result;
    goto LABEL_28;
  }
  while ( v40 != v26 )
  {
    v31 = *(_QWORD *)(v26 + 8);
    v26 += 32;
    j___libc_free_0(v31);
  }
  while ( v42 != v25 )
  {
    v32 = *(_QWORD *)(v25 + 8);
    v25 += 32;
    j___libc_free_0(v32);
  }
  result = a1;
  v34 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
LABEL_28:
    v35 = *(__int64 **)(a1 + 40);
    v36 = *(_QWORD *)(a1 + 72) + 8LL;
    if ( v36 > (unsigned __int64)v35 )
    {
      do
      {
        v37 = *v35++;
        j_j___libc_free_0(v37, 512);
      }
      while ( v36 > (unsigned __int64)v35 );
      v34 = *(_QWORD *)a1;
    }
    return j_j___libc_free_0(v34, 8LL * *(_QWORD *)(a1 + 8));
  }
  return result;
}
