// Function: sub_DB1930
// Address: 0xdb1930
//
__int64 __fastcall sub_DB1930(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4)
{
  const __m128i *v7; // rbx
  __int64 v8; // rax
  unsigned int v9; // r8d
  __m128i v10; // kr00_16
  __int64 v11; // r11
  unsigned int v12; // r10d
  __int64 v13; // rdx
  __int64 v14; // r15
  bool v15; // al
  int v16; // esi
  int v17; // ecx
  unsigned int v18; // esi
  __int64 v19; // rdi
  __int64 v20; // r9
  unsigned int v21; // ecx
  __int64 *v22; // r14
  _QWORD *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // rcx
  _QWORD *v27; // r12
  unsigned __int64 v28; // r8
  __m128i *v29; // rdx
  __int64 result; // rax
  unsigned int v31; // esi
  __int64 v32; // rdi
  __int64 v33; // r9
  int v34; // r11d
  unsigned int v35; // ecx
  __int64 *v36; // rdx
  _QWORD *v37; // rax
  __int64 v38; // r8
  __int64 v39; // rsi
  _QWORD *v40; // rax
  __m128i *v41; // r10
  __int64 v42; // r9
  __m128i *v43; // rcx
  __m128i *v44; // rsi
  int v45; // eax
  int v46; // esi
  __int64 v47; // r8
  unsigned int v48; // edx
  int v49; // ecx
  __int64 v50; // rdi
  int v51; // r10d
  _QWORD *v52; // r9
  int v53; // edi
  int v54; // edi
  int v55; // ecx
  __int64 v56; // rdx
  _QWORD *v57; // rdx
  const void *v58; // rsi
  __int8 *v59; // rbx
  __int64 v60; // rax
  int v61; // eax
  int v62; // eax
  int v63; // edx
  __int64 v64; // rdi
  _QWORD *v65; // r8
  unsigned int v66; // r15d
  int v67; // r9d
  __int64 v68; // rsi
  __int64 v69; // [rsp+0h] [rbp-B0h]
  unsigned int v70; // [rsp+1Ch] [rbp-94h]
  __int64 v71; // [rsp+20h] [rbp-90h]
  unsigned int v72; // [rsp+30h] [rbp-80h]
  int v73; // [rsp+34h] [rbp-7Ch]
  __int16 v74; // [rsp+3Ah] [rbp-76h]
  _BYTE v75[12]; // [rsp+3Ch] [rbp-74h] BYREF
  __int64 v76; // [rsp+48h] [rbp-68h]
  __int16 v77; // [rsp+50h] [rbp-60h]
  __m128i v78; // [rsp+60h] [rbp-50h] BYREF
  __int64 v79; // [rsp+70h] [rbp-40h]
  __int64 v80; // [rsp+78h] [rbp-38h]

  v7 = a1;
  v8 = a1[1].m128i_i64[0];
  v80 = a2;
  v9 = *(_DWORD *)(a3 + 24);
  v79 = v8;
  v78 = _mm_loadu_si128(a1);
  if ( !v9 )
  {
    ++*(_QWORD *)a3;
    *(_QWORD *)&v75[4] = 0;
    goto LABEL_9;
  }
  v10 = v78;
  v11 = *(_QWORD *)(a3 + 8);
  v77 = 1;
  v12 = v9 - 1;
  *(_DWORD *)&v75[8] = 0;
  v76 = 0;
  v74 = v79;
  v73 = 1;
  v13 = 0;
  *(_QWORD *)v75 = (v9 - 1)
                 & ((unsigned int)((0xBF58476D1CE4E5B9LL
                                  * (((unsigned __int64)(unsigned __int16)v79 << 32)
                                   | (484763065 * v78.m128i_i32[2])
                                   ^ (unsigned int)((0xBF58476D1CE4E5B9LL * (v78.m128i_u32[2] | (v78.m128i_i64[0] << 32))) >> 31))) >> 31)
                  ^ (484763065
                   * ((484763065 * v78.m128i_i32[2])
                    ^ (unsigned int)((0xBF58476D1CE4E5B9LL * (v78.m128i_u32[2] | (v78.m128i_i64[0] << 32))) >> 31))));
  while ( 1 )
  {
    v14 = v11 + 32LL * *(unsigned int *)v75;
    if ( *(_OWORD *)&v10 == *(_OWORD *)v14 && v74 == *(_WORD *)(v14 + 16) )
    {
      v31 = *(_DWORD *)(a4 + 24);
      if ( v31 )
      {
        v32 = *(_QWORD *)(v14 + 24);
        v33 = *(_QWORD *)(a4 + 8);
        v34 = 1;
        v35 = (v31 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
        v36 = (__int64 *)(v33 + 72LL * v35);
        v37 = 0;
        v38 = *v36;
        if ( v32 == *v36 )
        {
LABEL_20:
          v39 = *((unsigned int *)v36 + 4);
          v40 = v36 + 1;
          if ( (_DWORD)v39 )
          {
            v41 = (__m128i *)v36[1];
            v42 = (unsigned int)(v39 - 1);
            v43 = v41;
            while ( v43->m128i_i64[0] != v7->m128i_i64[0]
                 || v43->m128i_i64[1] != v7->m128i_i64[1]
                 || v43[1].m128i_i16[0] != v7[1].m128i_i16[0] )
            {
              v43 = (__m128i *)((char *)v43 + 24);
              if ( &v41[1].m128i_u64[3 * v42 + 1] == (unsigned __int64 *)v43 )
                goto LABEL_27;
            }
            v44 = (__m128i *)((char *)v41 + 24 * v39 - 24);
            v78 = _mm_loadu_si128(v43);
            v79 = v43[1].m128i_i64[0];
            *v43 = _mm_loadu_si128(v44);
            v43[1].m128i_i16[0] = v44[1].m128i_i16[0];
            *v44 = _mm_loadu_si128(&v78);
            v44[1].m128i_i16[0] = v79;
            LODWORD(v42) = *((_DWORD *)v36 + 4) - 1;
LABEL_27:
            *((_DWORD *)v40 + 2) = v42;
            *(_QWORD *)(v14 + 24) = a2;
            v18 = *(_DWORD *)(a4 + 24);
            if ( v18 )
              goto LABEL_14;
            goto LABEL_28;
          }
LABEL_62:
          LODWORD(v42) = -1;
          goto LABEL_27;
        }
        while ( v38 != -4096 )
        {
          if ( v38 == -8192 && !v37 )
            v37 = v36;
          v35 = (v31 - 1) & (v34 + v35);
          v36 = (__int64 *)(v33 + 72LL * v35);
          v38 = *v36;
          if ( v32 == *v36 )
            goto LABEL_20;
          ++v34;
        }
        v54 = *(_DWORD *)(a4 + 16);
        if ( !v37 )
          v37 = v36;
        ++*(_QWORD *)a4;
        v55 = v54 + 1;
        v78.m128i_i64[0] = (__int64)v37;
        if ( 4 * (v54 + 1) < 3 * v31 )
        {
          if ( v31 - *(_DWORD *)(a4 + 20) - v55 > v31 >> 3 )
          {
LABEL_59:
            *(_DWORD *)(a4 + 16) = v55;
            if ( *v37 != -4096 )
              --*(_DWORD *)(a4 + 20);
            v56 = *(_QWORD *)(v14 + 24);
            v37[2] = 0x200000000LL;
            *v37 = v56;
            v57 = v37 + 3;
            v40 = v37 + 1;
            *v40 = v57;
            goto LABEL_62;
          }
LABEL_86:
          sub_DA5BD0(a4, v31);
          sub_D9E340(a4, (__int64 *)(v14 + 24), &v78);
          v55 = *(_DWORD *)(a4 + 16) + 1;
          v37 = (_QWORD *)v78.m128i_i64[0];
          goto LABEL_59;
        }
      }
      else
      {
        ++*(_QWORD *)a4;
        v78.m128i_i64[0] = 0;
      }
      v31 *= 2;
      goto LABEL_86;
    }
    if ( !*(_QWORD *)v14 && !*(_QWORD *)(v14 + 8) && !*(_WORD *)(v14 + 16) )
      break;
    v69 = v13;
    v70 = v12;
    v72 = v9;
    v71 = v11;
    v15 = sub_D95440(v11 + 32LL * *(unsigned int *)v75, (__int64)&v75[4]);
    v11 = v71;
    v9 = v72;
    v12 = v70;
    if ( v69 || !v15 )
      v14 = v69;
    v13 = v14;
    *(_DWORD *)v75 = v70 & (v73 + *(_DWORD *)v75);
    ++v73;
  }
  v61 = *(_DWORD *)(a3 + 16);
  if ( !v13 )
    v13 = v11 + 32LL * *(unsigned int *)v75;
  ++*(_QWORD *)a3;
  v17 = v61 + 1;
  *(_QWORD *)&v75[4] = v13;
  if ( 4 * (v61 + 1) >= 3 * v9 )
  {
LABEL_9:
    v16 = 2 * v9;
    goto LABEL_10;
  }
  if ( v9 - *(_DWORD *)(a3 + 20) - v17 <= v9 >> 3 )
  {
    v16 = v9;
LABEL_10:
    sub_DB15D0(a3, v16);
    sub_D9F780(a3, v78.m128i_i64, &v75[4]);
    v13 = *(_QWORD *)&v75[4];
    v17 = *(_DWORD *)(a3 + 16) + 1;
  }
  *(_DWORD *)(a3 + 16) = v17;
  if ( *(_QWORD *)v13 || *(_QWORD *)(v13 + 8) || *(_WORD *)(v13 + 16) )
    --*(_DWORD *)(a3 + 20);
  *(__m128i *)v13 = _mm_loadu_si128(&v78);
  *(_WORD *)(v13 + 16) = v79;
  *(_QWORD *)(v13 + 24) = v80;
  v18 = *(_DWORD *)(a4 + 24);
  if ( !v18 )
  {
LABEL_28:
    ++*(_QWORD *)a4;
    goto LABEL_29;
  }
LABEL_14:
  v19 = *(_QWORD *)(a4 + 8);
  v20 = 1;
  v21 = (v18 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v22 = (__int64 *)(v19 + 72LL * v21);
  v23 = 0;
  v24 = *v22;
  if ( *v22 != a2 )
  {
    while ( v24 != -4096 )
    {
      if ( v24 == -8192 && !v23 )
        v23 = v22;
      v21 = (v18 - 1) & (v20 + v21);
      v22 = (__int64 *)(v19 + 72LL * v21);
      v24 = *v22;
      if ( *v22 == a2 )
        goto LABEL_15;
      v20 = (unsigned int)(v20 + 1);
    }
    v53 = *(_DWORD *)(a4 + 16);
    if ( !v23 )
      v23 = v22;
    ++*(_QWORD *)a4;
    v49 = v53 + 1;
    if ( 4 * (v53 + 1) < 3 * v18 )
    {
      if ( v18 - *(_DWORD *)(a4 + 20) - v49 > v18 >> 3 )
      {
LABEL_46:
        *(_DWORD *)(a4 + 16) = v49;
        if ( *v23 != -4096 )
          --*(_DWORD *)(a4 + 20);
        v29 = (__m128i *)(v23 + 3);
        *v23 = a2;
        v27 = v23 + 1;
        v23[1] = v23 + 3;
        v23[2] = 0x200000000LL;
        goto LABEL_16;
      }
      sub_DA5BD0(a4, v18);
      v62 = *(_DWORD *)(a4 + 24);
      if ( v62 )
      {
        v63 = v62 - 1;
        v64 = *(_QWORD *)(a4 + 8);
        v65 = 0;
        v66 = (v62 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v67 = 1;
        v49 = *(_DWORD *)(a4 + 16) + 1;
        v23 = (_QWORD *)(v64 + 72LL * v66);
        v68 = *v23;
        if ( *v23 != a2 )
        {
          while ( v68 != -4096 )
          {
            if ( v68 == -8192 && !v65 )
              v65 = v23;
            v66 = v63 & (v67 + v66);
            v23 = (_QWORD *)(v64 + 72LL * v66);
            v68 = *v23;
            if ( *v23 == a2 )
              goto LABEL_46;
            ++v67;
          }
          if ( v65 )
            v23 = v65;
        }
        goto LABEL_46;
      }
LABEL_97:
      ++*(_DWORD *)(a4 + 16);
      BUG();
    }
LABEL_29:
    sub_DA5BD0(a4, 2 * v18);
    v45 = *(_DWORD *)(a4 + 24);
    if ( v45 )
    {
      v46 = v45 - 1;
      v47 = *(_QWORD *)(a4 + 8);
      v48 = (v45 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v49 = *(_DWORD *)(a4 + 16) + 1;
      v23 = (_QWORD *)(v47 + 72LL * v48);
      v50 = *v23;
      if ( *v23 != a2 )
      {
        v51 = 1;
        v52 = 0;
        while ( v50 != -4096 )
        {
          if ( !v52 && v50 == -8192 )
            v52 = v23;
          v48 = v46 & (v51 + v48);
          v23 = (_QWORD *)(v47 + 72LL * v48);
          v50 = *v23;
          if ( *v23 == a2 )
            goto LABEL_46;
          ++v51;
        }
        if ( v52 )
          v23 = v52;
      }
      goto LABEL_46;
    }
    goto LABEL_97;
  }
LABEL_15:
  v25 = *((unsigned int *)v22 + 4);
  v26 = v22[1];
  v27 = v22 + 1;
  v28 = v25 + 1;
  v29 = (__m128i *)(v26 + 24 * v25);
  if ( v25 + 1 > (unsigned __int64)*((unsigned int *)v22 + 5) )
  {
    v58 = v22 + 3;
    if ( v26 > (unsigned __int64)v7 || v29 <= v7 )
    {
      sub_C8D5F0((__int64)(v22 + 1), v58, v28, 0x18u, v28, v20);
      v29 = (__m128i *)(v22[1] + 24LL * *((unsigned int *)v22 + 4));
    }
    else
    {
      v59 = &v7->m128i_i8[-v26];
      sub_C8D5F0((__int64)(v22 + 1), v58, v28, 0x18u, v28, v20);
      v60 = v22[1];
      v7 = (const __m128i *)&v59[v60];
      v29 = (__m128i *)(v60 + 24LL * *((unsigned int *)v22 + 4));
    }
  }
LABEL_16:
  *v29 = _mm_loadu_si128(v7);
  result = v7[1].m128i_i64[0];
  v29[1].m128i_i64[0] = result;
  ++*((_DWORD *)v27 + 2);
  return result;
}
