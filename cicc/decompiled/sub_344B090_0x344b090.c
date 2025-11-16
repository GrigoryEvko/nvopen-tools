// Function: sub_344B090
// Address: 0x344b090
//
__int64 __fastcall sub_344B090(
        unsigned int *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        __int128 a7,
        __int64 a8,
        __int64 a9)
{
  bool v10; // zf
  unsigned int *v11; // r9
  __int64 v13; // rax
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // r14
  char v20; // al
  __int64 v21; // rcx
  __int64 v22; // r8
  _QWORD *v23; // r9
  __m128i v24; // xmm0
  char v25; // al
  unsigned __int16 *v26; // rax
  __int64 v27; // r15
  unsigned int v28; // r14d
  __int128 v29; // rax
  unsigned __int8 *v30; // r14
  __int64 v31; // rdx
  __int64 v32; // r15
  __int128 v33; // rax
  __int128 v34; // [rsp-30h] [rbp-100h]
  __int64 v35; // [rsp+8h] [rbp-C8h]
  _QWORD *v36; // [rsp+8h] [rbp-C8h]
  unsigned int v37; // [rsp+3Ch] [rbp-94h] BYREF
  __m128i v38; // [rsp+40h] [rbp-90h] BYREF
  __int128 v39; // [rsp+50h] [rbp-80h] BYREF
  __int128 v40; // [rsp+60h] [rbp-70h] BYREF
  unsigned int *v41[12]; // [rsp+70h] [rbp-60h] BYREF

  v10 = *(_DWORD *)(a4 + 24) == 186;
  v38.m128i_i64[0] = 0;
  v38.m128i_i32[2] = 0;
  v11 = *(unsigned int **)(a8 + 16);
  *(_QWORD *)&v39 = 0;
  v41[0] = &v37;
  v41[1] = (unsigned int *)&v38;
  v41[2] = (unsigned int *)&v39;
  DWORD2(v39) = 0;
  *(_QWORD *)&v40 = 0;
  DWORD2(v40) = 0;
  v41[3] = (unsigned int *)&v40;
  v41[4] = v11;
  v41[5] = a1;
  if ( !v10 )
    return 0;
  v13 = *(_QWORD *)(a4 + 56);
  if ( !v13 )
    return 0;
  v15 = 1;
  do
  {
    while ( (_DWORD)a5 != *(_DWORD *)(v13 + 8) )
    {
      v13 = *(_QWORD *)(v13 + 32);
      if ( !v13 )
        goto LABEL_11;
    }
    if ( !v15 )
      return 0;
    v16 = *(_QWORD *)(v13 + 32);
    if ( !v16 )
      goto LABEL_12;
    if ( (_DWORD)a5 == *(_DWORD *)(v16 + 8) )
      return 0;
    v13 = *(_QWORD *)(v16 + 32);
    v15 = 0;
  }
  while ( v13 );
LABEL_11:
  if ( v15 == 1 )
    return 0;
LABEL_12:
  v17 = *(_QWORD *)(a4 + 40);
  v35 = (__int64)v11;
  v18 = *(_QWORD *)(v17 + 40);
  v19 = *(_QWORD *)(v17 + 48);
  v38.m128i_i64[0] = *(_QWORD *)v17;
  v38.m128i_i32[2] = *(_DWORD *)(v17 + 8);
  v20 = sub_3441C70(v41, v18, v19, a4, a5, (__int64)v11);
  v23 = (_QWORD *)v35;
  if ( !v20 )
  {
    v24 = _mm_loadu_si128(&v38);
    v38.m128i_i64[0] = v18;
    v38.m128i_i32[2] = v19;
    v25 = sub_3441C70(v41, v24.m128i_i64[0], v24.m128i_i32[2], v21, v22, v35);
    v23 = (_QWORD *)v35;
    if ( !v25 )
      return 0;
  }
  v36 = v23;
  v26 = (unsigned __int16 *)(*(_QWORD *)(v38.m128i_i64[0] + 48) + 16LL * v38.m128i_u32[2]);
  v27 = *((_QWORD *)v26 + 1);
  v28 = *v26;
  *(_QWORD *)&v29 = sub_3406EB0(v23, v37, a9, *v26, v27, (__int64)v23, *(_OWORD *)&v38, v40);
  v30 = sub_3406EB0(v36, 0xBAu, a9, v28, v27, (__int64)v36, v29, v39);
  v32 = v31;
  *(_QWORD *)&v33 = sub_33ED040(v36, a6);
  *((_QWORD *)&v34 + 1) = v32;
  *(_QWORD *)&v34 = v30;
  return sub_340F900(v36, 0xD0u, a9, a2, a3, (__int64)v36, v34, a7, v33);
}
