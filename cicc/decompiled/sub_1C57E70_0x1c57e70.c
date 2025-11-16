// Function: sub_1C57E70
// Address: 0x1c57e70
//
__int64 __fastcall sub_1C57E70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 *a9,
        __int64 a10)
{
  __int64 i; // r13
  __int64 v13; // r9
  __int64 v14; // rdi
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // r11
  unsigned int v18; // r11d
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r10
  __m128i *v22; // r15
  __int64 v23; // r12
  __m128i *v24; // r13
  unsigned int v25; // esi
  __int64 *v26; // rcx
  __int64 *v27; // r8
  __int64 *v28; // r10
  int v29; // edx
  __int64 v30; // rax
  __int64 *v31; // r9
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // r11
  __int64 v35; // r12
  unsigned int v36; // r8d
  __int64 v37; // rdi
  unsigned int v38; // edx
  __int64 *v39; // rax
  __int64 v40; // r10
  unsigned int v41; // r9d
  unsigned int v42; // edx
  __int64 *v43; // rax
  __int64 v44; // r10
  __m128i *v45; // r11
  unsigned int v46; // esi
  __int64 *v47; // r13
  __int64 *v48; // r9
  const __m128i *v49; // rax
  __int64 *v50; // r13
  __int64 *v51; // r9
  int v52; // eax
  int v53; // eax
  __int64 v54; // rax
  int v56; // eax
  __int64 v57; // rax
  int v58; // eax
  int v59; // eax
  __int64 *v60; // [rsp+10h] [rbp-80h]
  int v61; // [rsp+10h] [rbp-80h]
  int v62; // [rsp+10h] [rbp-80h]
  __int64 *v63; // [rsp+10h] [rbp-80h]
  __int64 v65; // [rsp+20h] [rbp-70h]
  int v66; // [rsp+20h] [rbp-70h]
  int v67; // [rsp+20h] [rbp-70h]
  __int64 v68; // [rsp+28h] [rbp-68h]
  __int64 v69; // [rsp+28h] [rbp-68h]
  __int64 v71; // [rsp+30h] [rbp-60h]
  unsigned int v72; // [rsp+38h] [rbp-58h]
  __int64 *v73; // [rsp+38h] [rbp-58h]
  __int64 v74; // [rsp+38h] [rbp-58h]
  __int64 *v75; // [rsp+38h] [rbp-58h]
  __int64 v76; // [rsp+40h] [rbp-50h]
  __int64 *v77; // [rsp+48h] [rbp-48h]
  __int64 *v78; // [rsp+48h] [rbp-48h]
  _QWORD v79[7]; // [rsp+58h] [rbp-38h] BYREF

  v68 = (a3 - 1) / 2;
  if ( a2 >= v68 )
  {
    v22 = (__m128i *)(a1 + 32 * a2);
    if ( (a3 & 1) != 0 )
    {
      v74 = a7;
      v76 = a8;
      v78 = a9;
      v69 = a10;
      goto LABEL_48;
    }
    v34 = a2;
    goto LABEL_37;
  }
  for ( i = a2; ; i = v23 )
  {
    v25 = *(_DWORD *)(a4 + 24);
    v23 = 2 * (i + 1);
    v22 = (__m128i *)(a1 + ((i + 1) << 6));
    v26 = (__int64 *)v22[1].m128i_i64[0];
    v27 = *(__int64 **)(a1 + 32 * (v23 - 1) + 16);
    if ( v25 )
    {
      v13 = *v26;
      v14 = *(_QWORD *)(a4 + 8);
      v72 = v25 - 1;
      v15 = (v25 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v16 = (__int64 *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( *v26 == *v16 )
      {
LABEL_4:
        v18 = *((_DWORD *)v16 + 2);
        goto LABEL_5;
      }
      v62 = 1;
      v28 = 0;
      while ( v17 != -8 )
      {
        if ( v17 == -16 && !v28 )
          v28 = v16;
        v15 = v72 & (v62 + v15);
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( v13 == *v16 )
          goto LABEL_4;
        ++v62;
      }
      if ( !v28 )
        v28 = v16;
      v59 = *(_DWORD *)(a4 + 16);
      ++*(_QWORD *)a4;
      v29 = v59 + 1;
      if ( 4 * (v59 + 1) < 3 * v25 )
      {
        if ( v25 - *(_DWORD *)(a4 + 20) - v29 <= v25 >> 3 )
        {
          v63 = v27;
          v75 = v26;
          sub_1468630(a4, v25);
          sub_145FB10(a4, v75, v79);
          v28 = (__int64 *)v79[0];
          v27 = v63;
          v26 = v75;
          v29 = *(_DWORD *)(a4 + 16) + 1;
        }
        goto LABEL_13;
      }
    }
    else
    {
      ++*(_QWORD *)a4;
    }
    v60 = v27;
    v73 = v26;
    sub_1468630(a4, 2 * v25);
    sub_145FB10(a4, v73, v79);
    v28 = (__int64 *)v79[0];
    v26 = v73;
    v27 = v60;
    v29 = *(_DWORD *)(a4 + 16) + 1;
LABEL_13:
    *(_DWORD *)(a4 + 16) = v29;
    if ( *v28 != -8 )
      --*(_DWORD *)(a4 + 20);
    v30 = *v26;
    *((_DWORD *)v28 + 2) = 0;
    *v28 = v30;
    v25 = *(_DWORD *)(a4 + 24);
    if ( !v25 )
    {
      ++*(_QWORD *)a4;
      goto LABEL_17;
    }
    v14 = *(_QWORD *)(a4 + 8);
    v18 = 0;
    v72 = v25 - 1;
LABEL_5:
    v19 = v72 & (((unsigned int)*v27 >> 9) ^ ((unsigned int)*v27 >> 4));
    v20 = (__int64 *)(v14 + 16LL * v19);
    v21 = *v20;
    if ( *v27 != *v20 )
    {
      v61 = 1;
      v31 = 0;
      while ( v21 != -8 )
      {
        if ( !v31 && v21 == -16 )
          v31 = v20;
        v19 = v72 & (v61 + v19);
        v20 = (__int64 *)(v14 + 16LL * v19);
        v21 = *v20;
        if ( *v27 == *v20 )
          goto LABEL_6;
        ++v61;
      }
      if ( !v31 )
        v31 = v20;
      v58 = *(_DWORD *)(a4 + 16);
      ++*(_QWORD *)a4;
      v32 = v58 + 1;
      if ( 4 * v32 < 3 * v25 )
      {
        if ( v25 - (v32 + *(_DWORD *)(a4 + 20)) > v25 >> 3 )
          goto LABEL_19;
        v77 = v27;
LABEL_18:
        sub_1468630(a4, v25);
        sub_145FB10(a4, v77, v79);
        v31 = (__int64 *)v79[0];
        v27 = v77;
        v32 = *(_DWORD *)(a4 + 16) + 1;
LABEL_19:
        *(_DWORD *)(a4 + 16) = v32;
        if ( *v31 != -8 )
          --*(_DWORD *)(a4 + 20);
        v33 = *v27;
        *((_DWORD *)v31 + 2) = 0;
        *v31 = v33;
        goto LABEL_8;
      }
LABEL_17:
      v77 = v27;
      v25 *= 2;
      goto LABEL_18;
    }
LABEL_6:
    if ( *((_DWORD *)v20 + 2) > v18 )
      v22 = (__m128i *)(a1 + 32 * --v23);
LABEL_8:
    v24 = (__m128i *)(a1 + 32 * i);
    *v24 = _mm_loadu_si128(v22);
    v24[1] = _mm_loadu_si128(v22 + 1);
    if ( v23 >= v68 )
      break;
  }
  v34 = v23;
  if ( (a3 & 1) == 0 )
  {
LABEL_37:
    if ( (a3 - 2) / 2 == v34 )
    {
      v34 = 2 * v34 + 1;
      v49 = (const __m128i *)(a1 + 32 * v34);
      *v22 = _mm_loadu_si128(v49);
      v22[1] = _mm_loadu_si128(v49 + 1);
      v22 = (__m128i *)v49;
    }
  }
  v74 = a7;
  v76 = a8;
  v78 = a9;
  v69 = a10;
  v35 = (v34 - 1) / 2;
  if ( v34 <= a2 )
    goto LABEL_48;
  while ( 2 )
  {
    v46 = *(_DWORD *)(a4 + 24);
    v22 = (__m128i *)(a1 + 32 * v35);
    v47 = (__int64 *)v22[1].m128i_i64[0];
    if ( !v46 )
    {
      ++*(_QWORD *)a4;
      goto LABEL_33;
    }
    v36 = v46 - 1;
    v37 = *(_QWORD *)(a4 + 8);
    v38 = (v46 - 1) & (((unsigned int)*v47 >> 9) ^ ((unsigned int)*v47 >> 4));
    v39 = (__int64 *)(v37 + 16LL * v38);
    v40 = *v39;
    if ( *v39 == *v47 )
    {
LABEL_26:
      v41 = *((_DWORD *)v39 + 2);
      goto LABEL_27;
    }
    v67 = 1;
    v48 = 0;
    while ( v40 != -8 )
    {
      if ( v40 == -16 && !v48 )
        v48 = v39;
      v38 = v36 & (v67 + v38);
      v39 = (__int64 *)(v37 + 16LL * v38);
      v40 = *v39;
      if ( *v47 == *v39 )
        goto LABEL_26;
      ++v67;
    }
    if ( !v48 )
      v48 = v39;
    v56 = *(_DWORD *)(a4 + 16);
    ++*(_QWORD *)a4;
    if ( 4 * (v56 + 1) >= 3 * v46 )
    {
LABEL_33:
      v65 = v34;
      v46 *= 2;
    }
    else
    {
      if ( v46 - *(_DWORD *)(a4 + 20) - (v56 + 1) > v46 >> 3 )
        goto LABEL_55;
      v65 = v34;
    }
    sub_1468630(a4, v46);
    sub_145FB10(a4, v47, v79);
    v48 = (__int64 *)v79[0];
    v34 = v65;
LABEL_55:
    ++*(_DWORD *)(a4 + 16);
    if ( *v48 != -8 )
      --*(_DWORD *)(a4 + 20);
    v57 = *v47;
    *((_DWORD *)v48 + 2) = 0;
    *v48 = v57;
    v46 = *(_DWORD *)(a4 + 24);
    if ( !v46 )
    {
      ++*(_QWORD *)a4;
LABEL_59:
      v71 = v34;
      v46 *= 2;
      goto LABEL_60;
    }
    v37 = *(_QWORD *)(a4 + 8);
    v36 = v46 - 1;
    v41 = 0;
LABEL_27:
    v42 = v36 & (((unsigned int)*a9 >> 9) ^ ((unsigned int)*a9 >> 4));
    v43 = (__int64 *)(v37 + 16LL * v42);
    v44 = *v43;
    if ( *v43 == *a9 )
    {
LABEL_28:
      v45 = (__m128i *)(a1 + 32 * v34);
      if ( *((_DWORD *)v43 + 2) <= v41 )
      {
        v22 = v45;
        goto LABEL_48;
      }
      *v45 = _mm_loadu_si128(v22);
      v45[1] = _mm_loadu_si128(v22 + 1);
      v34 = v35;
      if ( a2 >= v35 )
        goto LABEL_48;
      v35 = (v35 - 1) / 2;
      continue;
    }
    break;
  }
  v66 = 1;
  v50 = 0;
  while ( v44 != -8 )
  {
    if ( !v50 && v44 == -16 )
      v50 = v43;
    v42 = v36 & (v66 + v42);
    v43 = (__int64 *)(v37 + 16LL * v42);
    v44 = *v43;
    if ( *a9 == *v43 )
      goto LABEL_28;
    ++v66;
  }
  v51 = v50;
  if ( !v50 )
    v51 = v43;
  v52 = *(_DWORD *)(a4 + 16);
  ++*(_QWORD *)a4;
  v53 = v52 + 1;
  if ( 4 * v53 >= 3 * v46 )
    goto LABEL_59;
  if ( v46 - (*(_DWORD *)(a4 + 20) + v53) > v46 >> 3 )
    goto LABEL_45;
  v71 = v34;
LABEL_60:
  sub_1468630(a4, v46);
  sub_145FB10(a4, a9, v79);
  v51 = (__int64 *)v79[0];
  v34 = v71;
LABEL_45:
  ++*(_DWORD *)(a4 + 16);
  if ( *v51 != -8 )
    --*(_DWORD *)(a4 + 20);
  v54 = *a9;
  *((_DWORD *)v51 + 2) = 0;
  *v51 = v54;
  v22 = (__m128i *)(a1 + 32 * v34);
LABEL_48:
  v22->m128i_i64[0] = v74;
  v22->m128i_i64[1] = v76;
  v22[1].m128i_i64[0] = (__int64)v78;
  v22[1].m128i_i64[1] = v69;
  return v69;
}
