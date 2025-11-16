// Function: sub_FF2910
// Address: 0xff2910
//
__int64 __fastcall sub_FF2910(__int64 a1, __int64 *a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rbx
  char v8; // si
  __int64 v9; // r8
  int v10; // ecx
  unsigned int v11; // edx
  _QWORD *v12; // rax
  __int64 v13; // r10
  unsigned int v15; // ecx
  unsigned int v16; // eax
  _QWORD *v17; // rdi
  int v18; // edx
  unsigned int v19; // esi
  __int64 v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // r13
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdi
  int v26; // ecx
  int v27; // r10d
  unsigned int i; // eax
  __int64 v29; // r8
  unsigned int v30; // eax
  __int64 v31; // rsi
  int v32; // ecx
  unsigned int v33; // eax
  __int64 v34; // rdx
  int v35; // ecx
  int v36; // ecx
  const __m128i *v37; // rdx
  __int64 v38; // rax
  unsigned __int64 v39; // rcx
  unsigned __int64 v40; // r8
  __m128i *v41; // rax
  int v42; // edi
  __int64 v43; // rax
  int v44; // r11d
  const void *v45; // rsi
  _BYTE *v46; // r13
  __int64 v47; // r8
  int v48; // edx
  __int64 v49; // rax
  __int64 v50; // r11
  __int64 v51; // r8
  int v52; // edx
  __int64 v53; // rax
  __int64 v54; // r11
  int v55; // esi
  _QWORD *v56; // rcx
  int v57; // edx
  int v58; // edx
  int v59; // esi
  __int64 v63[2]; // [rsp+20h] [rbp-60h] BYREF
  _BYTE v64[8]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v65; // [rsp+38h] [rbp-48h]
  int v66; // [rsp+40h] [rbp-40h]

  v7 = *a2;
  v8 = *(_BYTE *)(a1 + 96) & 1;
  if ( v8 )
  {
    v9 = a1 + 104;
    v10 = 3;
  }
  else
  {
    v15 = *(_DWORD *)(a1 + 112);
    v9 = *(_QWORD *)(a1 + 104);
    if ( !v15 )
    {
      v16 = *(_DWORD *)(a1 + 96);
      ++*(_QWORD *)(a1 + 88);
      v17 = 0;
      v18 = (v16 >> 1) + 1;
LABEL_8:
      v19 = 3 * v15;
      goto LABEL_9;
    }
    v10 = v15 - 1;
  }
  v11 = v10 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v12 = (_QWORD *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( v7 == *v12 )
    return 0;
  v44 = 1;
  v17 = 0;
  while ( v13 != -4096 )
  {
    if ( v13 == -8192 && !v17 )
      v17 = v12;
    v11 = v10 & (v44 + v11);
    v12 = (_QWORD *)(v9 + 16LL * v11);
    v13 = *v12;
    if ( v7 == *v12 )
      return 0;
    ++v44;
  }
  if ( !v17 )
    v17 = v12;
  v16 = *(_DWORD *)(a1 + 96);
  ++*(_QWORD *)(a1 + 88);
  v18 = (v16 >> 1) + 1;
  if ( !v8 )
  {
    v15 = *(_DWORD *)(a1 + 112);
    goto LABEL_8;
  }
  v19 = 12;
  v15 = 4;
LABEL_9:
  if ( 4 * v18 < v19 )
  {
    if ( v15 - *(_DWORD *)(a1 + 100) - v18 > v15 >> 3 )
      goto LABEL_11;
    sub_FF24F0(a1 + 88, v15);
    if ( (*(_BYTE *)(a1 + 96) & 1) != 0 )
    {
      v51 = a1 + 104;
      v52 = 3;
      goto LABEL_62;
    }
    v58 = *(_DWORD *)(a1 + 112);
    v51 = *(_QWORD *)(a1 + 104);
    if ( v58 )
    {
      v52 = v58 - 1;
LABEL_62:
      v53 = v52 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v17 = (_QWORD *)(v51 + 16 * v53);
      v54 = *v17;
      if ( v7 != *v17 )
      {
        v55 = 1;
        v56 = 0;
        while ( v54 != -4096 )
        {
          if ( v54 == -8192 && !v56 )
            v56 = v17;
          LODWORD(v53) = v52 & (v55 + v53);
          v17 = (_QWORD *)(v51 + 16LL * (unsigned int)v53);
          v54 = *v17;
          if ( v7 == *v17 )
            goto LABEL_59;
          ++v55;
        }
LABEL_65:
        if ( v56 )
          v17 = v56;
        goto LABEL_59;
      }
      goto LABEL_59;
    }
LABEL_89:
    *(_DWORD *)(a1 + 96) = (2 * (*(_DWORD *)(a1 + 96) >> 1) + 2) | *(_DWORD *)(a1 + 96) & 1;
    BUG();
  }
  sub_FF24F0(a1 + 88, 2 * v15);
  if ( (*(_BYTE *)(a1 + 96) & 1) != 0 )
  {
    v47 = a1 + 104;
    v48 = 3;
  }
  else
  {
    v57 = *(_DWORD *)(a1 + 112);
    v47 = *(_QWORD *)(a1 + 104);
    if ( !v57 )
      goto LABEL_89;
    v48 = v57 - 1;
  }
  v49 = v48 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v17 = (_QWORD *)(v47 + 16 * v49);
  v50 = *v17;
  if ( v7 != *v17 )
  {
    v59 = 1;
    v56 = 0;
    while ( v50 != -4096 )
    {
      if ( !v56 && v50 == -8192 )
        v56 = v17;
      LODWORD(v49) = v48 & (v59 + v49);
      v17 = (_QWORD *)(v47 + 16LL * (unsigned int)v49);
      v50 = *v17;
      if ( v7 == *v17 )
        goto LABEL_59;
      ++v59;
    }
    goto LABEL_65;
  }
LABEL_59:
  v16 = *(_DWORD *)(a1 + 96);
LABEL_11:
  *(_DWORD *)(a1 + 96) = (2 * (v16 >> 1) + 2) | v16 & 1;
  if ( *v17 != -4096 )
    --*(_DWORD *)(a1 + 100);
  *v17 = v7;
  *((_DWORD *)v17 + 2) = a3;
  v20 = *(_QWORD *)(v7 + 16);
  if ( v20 )
  {
    while ( 1 )
    {
      v21 = *(_QWORD *)(v20 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v21 - 30) <= 0xAu )
        break;
      v20 = *(_QWORD *)(v20 + 8);
      if ( !v20 )
        return 1;
    }
LABEL_15:
    v22 = *(_QWORD *)(v21 + 40);
    sub_FEF2D0((__int64)v64, v22, *(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80));
    v63[0] = (__int64)v64;
    v63[1] = (__int64)a2;
    if ( sub_FEF3D0(a1, v63) )
    {
      if ( (*(_BYTE *)(a1 + 176) & 1) != 0 )
      {
        v25 = a1 + 184;
        v26 = 3;
      }
      else
      {
        v35 = *(_DWORD *)(a1 + 192);
        v25 = *(_QWORD *)(a1 + 184);
        if ( !v35 )
        {
LABEL_35:
          v37 = (const __m128i *)v64;
          v38 = *(unsigned int *)(a5 + 8);
          v39 = *(_QWORD *)a5;
          v40 = v38 + 1;
          if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
          {
            v45 = (const void *)(a5 + 16);
            if ( v39 > (unsigned __int64)v64 || (unsigned __int64)v64 >= v39 + 24 * v38 )
            {
              sub_C8D5F0(a5, v45, v40, 0x18u, v40, v24);
              v39 = *(_QWORD *)a5;
              v38 = *(unsigned int *)(a5 + 8);
              v37 = (const __m128i *)v64;
            }
            else
            {
              v46 = &v64[-v39];
              sub_C8D5F0(a5, v45, v40, 0x18u, v40, v24);
              v39 = *(_QWORD *)a5;
              v38 = *(unsigned int *)(a5 + 8);
              v37 = (const __m128i *)&v46[*(_QWORD *)a5];
            }
          }
          v41 = (__m128i *)(v39 + 24 * v38);
          *v41 = _mm_loadu_si128(v37);
          v41[1].m128i_i64[0] = v37[1].m128i_i64[0];
          ++*(_DWORD *)(a5 + 8);
          goto LABEL_27;
        }
        v26 = v35 - 1;
      }
      v27 = 1;
      for ( i = v26
              & (((0xBF58476D1CE4E5B9LL
                 * ((unsigned int)(37 * v66)
                  | ((unsigned __int64)(((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4)) << 32))) >> 31)
               ^ (756364221 * v66)); ; i = v26 & v30 )
      {
        v29 = v25 + 24LL * i;
        v24 = *(_QWORD *)v29;
        if ( v65 == *(_QWORD *)v29 && v66 == *(_DWORD *)(v29 + 8) )
          goto LABEL_27;
        if ( v24 == -4096 && *(_DWORD *)(v29 + 8) == 0x7FFFFFFF )
          break;
        v30 = v27 + i;
        ++v27;
      }
      goto LABEL_35;
    }
    if ( (*(_BYTE *)(a1 + 96) & 1) != 0 )
    {
      v31 = a1 + 104;
      v32 = 3;
    }
    else
    {
      v36 = *(_DWORD *)(a1 + 112);
      v31 = *(_QWORD *)(a1 + 104);
      if ( !v36 )
        goto LABEL_39;
      v32 = v36 - 1;
    }
    v33 = v32 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v34 = *(_QWORD *)(v31 + 16LL * v33);
    if ( v22 == v34 )
      goto LABEL_27;
    v42 = 1;
    while ( v34 != -4096 )
    {
      v23 = (unsigned int)(v42 + 1);
      v33 = v32 & (v42 + v33);
      v34 = *(_QWORD *)(v31 + 16LL * v33);
      if ( v22 == v34 )
        goto LABEL_27;
      ++v42;
    }
LABEL_39:
    v43 = *(unsigned int *)(a4 + 8);
    if ( v43 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
      sub_C8D5F0(a4, (const void *)(a4 + 16), v43 + 1, 8u, v23, v24);
      v43 = *(unsigned int *)(a4 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a4 + 8 * v43) = v22;
    ++*(_DWORD *)(a4 + 8);
LABEL_27:
    while ( 1 )
    {
      v20 = *(_QWORD *)(v20 + 8);
      if ( !v20 )
        break;
      v21 = *(_QWORD *)(v20 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v21 - 30) <= 0xAu )
        goto LABEL_15;
    }
  }
  return 1;
}
