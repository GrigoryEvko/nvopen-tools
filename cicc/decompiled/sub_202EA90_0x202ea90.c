// Function: sub_202EA90
// Address: 0x202ea90
//
__int64 *__fastcall sub_202EA90(__int64 *a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  __int64 v7; // rsi
  char *v8; // rax
  char v9; // dl
  unsigned int *v10; // rax
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rdx
  const __m128i *v14; // rax
  int v15; // esi
  unsigned __int64 v16; // rcx
  const __m128i *v17; // r8
  unsigned __int64 v18; // r9
  __m128 *v19; // rdx
  _QWORD *v20; // rdi
  _QWORD *v21; // rax
  int v22; // edx
  int v23; // r15d
  __int64 v24; // rdx
  int v25; // r8d
  unsigned __int64 v26; // r9
  int v27; // ecx
  unsigned __int64 v28; // rsi
  _QWORD *v29; // rdx
  __int64 *v30; // rdi
  __int64 *v31; // r14
  __int128 v33; // [rsp-10h] [rbp-1C0h]
  const __m128i *v34; // [rsp+0h] [rbp-1B0h]
  const __m128i *v35; // [rsp+8h] [rbp-1A8h]
  _QWORD *v36; // [rsp+10h] [rbp-1A0h]
  int v37; // [rsp+10h] [rbp-1A0h]
  int v38; // [rsp+18h] [rbp-198h]
  _QWORD *v39; // [rsp+18h] [rbp-198h]
  int v40; // [rsp+24h] [rbp-18Ch]
  int v41; // [rsp+24h] [rbp-18Ch]
  unsigned __int8 v42; // [rsp+28h] [rbp-188h]
  unsigned __int64 v43; // [rsp+28h] [rbp-188h]
  __int64 v44; // [rsp+30h] [rbp-180h] BYREF
  int v45; // [rsp+38h] [rbp-178h]
  __int64 v46; // [rsp+40h] [rbp-170h] BYREF
  __int64 v47; // [rsp+48h] [rbp-168h]
  __int64 v48; // [rsp+50h] [rbp-160h] BYREF
  const void **v49; // [rsp+58h] [rbp-158h]
  __int64 v50; // [rsp+60h] [rbp-150h] BYREF
  int v51; // [rsp+68h] [rbp-148h]
  _QWORD *v52; // [rsp+70h] [rbp-140h] BYREF
  __int64 v53; // [rsp+78h] [rbp-138h]
  _QWORD v54[38]; // [rsp+80h] [rbp-130h] BYREF

  v7 = *(_QWORD *)(a2 + 72);
  v44 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v44, v7, 2);
  v45 = *(_DWORD *)(a2 + 64);
  v8 = *(char **)(a2 + 40);
  v9 = *v8;
  v47 = *((_QWORD *)v8 + 1);
  v10 = *(unsigned int **)(a2 + 32);
  LOBYTE(v46) = v9;
  v11 = *(_QWORD *)(*(_QWORD *)v10 + 40LL) + 16LL * v10[2];
  v12 = *(_QWORD *)(v11 + 8);
  v42 = *(_BYTE *)v11;
  if ( v9 )
    v38 = word_4305480[(unsigned __int8)(v9 - 14)];
  else
    v38 = sub_1F58D30((__int64)&v46);
  sub_1F40D10((__int64)&v52, *a1, *(_QWORD *)(a1[1] + 48), v46, v47);
  LOBYTE(v48) = v53;
  v49 = (const void **)v54[0];
  if ( (_BYTE)v53 )
    v40 = word_4305480[(unsigned __int8)(v53 - 14)];
  else
    v40 = sub_1F58D30((__int64)&v48);
  v13 = *(unsigned int *)(a2 + 56);
  v14 = *(const __m128i **)(a2 + 32);
  v15 = 0;
  v52 = v54;
  v53 = 0x1000000000LL;
  v16 = 40 * v13;
  v17 = (const __m128i *)((char *)v14 + 40 * v13);
  v18 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v13) >> 3);
  v19 = (__m128 *)v54;
  if ( v16 > 0x280 )
  {
    v34 = v14;
    v35 = v17;
    v37 = v18;
    sub_16CD150((__int64)&v52, v54, v18, 16, (int)v17, v18);
    v15 = v53;
    v14 = v34;
    v17 = v35;
    LODWORD(v18) = v37;
    v19 = (__m128 *)&v52[2 * (unsigned int)v53];
  }
  if ( v14 != v17 )
  {
    do
    {
      if ( v19 )
      {
        a3 = (__m128)_mm_loadu_si128(v14);
        *v19 = a3;
      }
      v14 = (const __m128i *)((char *)v14 + 40);
      ++v19;
    }
    while ( v17 != v14 );
    v15 = v53;
  }
  v20 = (_QWORD *)a1[1];
  LODWORD(v53) = v15 + v18;
  v50 = 0;
  v51 = 0;
  v21 = sub_1D2B300(v20, 0x30u, (__int64)&v50, v42, v12, (__int64)&v50);
  v23 = v22;
  if ( v50 )
  {
    v36 = v21;
    sub_161E7C0((__int64)&v50, v50);
    v21 = v36;
  }
  v24 = (unsigned int)v53;
  v25 = v40 - v38;
  v26 = (unsigned int)(v40 - v38);
  v27 = v53;
  if ( v26 > HIDWORD(v53) - (unsigned __int64)(unsigned int)v53 )
  {
    v39 = v21;
    v41 = v25;
    v43 = v26;
    sub_16CD150((__int64)&v52, v54, v26 + (unsigned int)v53, 16, v25, v26);
    v24 = (unsigned int)v53;
    v21 = v39;
    v25 = v41;
    v26 = v43;
    v27 = v53;
  }
  v28 = (unsigned __int64)v52;
  v29 = &v52[2 * v24];
  if ( v26 )
  {
    do
    {
      if ( v29 )
      {
        *v29 = v21;
        *((_DWORD *)v29 + 2) = v23;
      }
      v29 += 2;
      --v26;
    }
    while ( v26 );
    v28 = (unsigned __int64)v52;
    v27 = v53;
  }
  v30 = (__int64 *)a1[1];
  LODWORD(v53) = v27 + v25;
  *((_QWORD *)&v33 + 1) = (unsigned int)(v27 + v25);
  *(_QWORD *)&v33 = v28;
  v31 = sub_1D359D0(v30, 104, (__int64)&v44, v48, v49, 0, *(double *)a3.m128_u64, a4, a5, v33);
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
  if ( v44 )
    sub_161E7C0((__int64)&v44, v44);
  return v31;
}
