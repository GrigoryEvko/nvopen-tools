// Function: sub_2F78160
// Address: 0x2f78160
//
__int64 __fastcall sub_2F78160(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // r14
  unsigned int v11; // r12d
  __int64 v12; // rbx
  __int64 v13; // r13
  unsigned int v14; // edi
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // eax
  __int64 v18; // r11
  __int64 v19; // r10
  __int64 v20; // rbx
  __int64 result; // rax
  unsigned int v22; // r12d
  __int64 v23; // rdi
  __int64 v24; // r14
  __int64 v25; // rdx
  __int64 v26; // rbx
  __int64 j; // r13
  int v28; // r12d
  unsigned int v29; // esi
  __int64 v30; // rcx
  unsigned int v31; // eax
  __int64 v32; // rdx
  _BYTE *v33; // r9
  unsigned int v34; // edi
  unsigned int v35; // eax
  _DWORD *v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  char v40; // r9
  __int64 v41; // rdi
  __int64 (*v42)(); // rax
  __m128i *v43; // rax
  __int64 v44; // r10
  signed __int64 v45; // r10
  __m128i *v46; // r10
  unsigned __int64 v47; // rsi
  __m128i *v48; // rbx
  int v49; // r8d
  __m128i *v50; // rdi
  signed __int64 v51; // rcx
  __m128i *v52; // rdx
  __m128i *v53; // rcx
  __int128 v54; // rax
  __int64 v55; // rdi
  unsigned __int64 v56; // rax
  __int64 i; // rsi
  int v58; // edx
  unsigned int v59; // edi
  unsigned __int64 v60; // rcx
  int v61; // edx
  int v62; // r10d
  unsigned __int64 v63; // r8
  const __m128i *v64; // rdx
  const void *v65; // rsi
  __int64 v66; // [rsp+8h] [rbp-A8h]
  __int64 v67; // [rsp+10h] [rbp-A0h]
  __int64 v68; // [rsp+10h] [rbp-A0h]
  __int64 v70; // [rsp+28h] [rbp-88h]
  __int64 v71; // [rsp+28h] [rbp-88h]
  __int64 v72; // [rsp+28h] [rbp-88h]
  __int64 v73; // [rsp+28h] [rbp-88h]
  __int64 v74; // [rsp+28h] [rbp-88h]
  __int64 v75; // [rsp+30h] [rbp-80h]
  __int64 v76; // [rsp+30h] [rbp-80h]
  __int64 v77; // [rsp+30h] [rbp-80h]
  __int64 v78; // [rsp+30h] [rbp-80h]
  __int64 v80; // [rsp+48h] [rbp-68h]
  unsigned __int64 v81; // [rsp+48h] [rbp-68h]
  __int64 v82; // [rsp+48h] [rbp-68h]
  __int64 v83; // [rsp+50h] [rbp-60h]
  char *v84; // [rsp+50h] [rbp-60h]
  __int64 v85; // [rsp+50h] [rbp-60h]
  __int64 v86; // [rsp+50h] [rbp-60h]
  __int64 v87; // [rsp+50h] [rbp-60h]
  __int64 v88; // [rsp+50h] [rbp-60h]
  __int64 v89; // [rsp+58h] [rbp-58h]
  __int64 v90; // [rsp+58h] [rbp-58h]
  __int64 v91; // [rsp+60h] [rbp-50h] BYREF
  __int128 v92; // [rsp+68h] [rbp-48h]

  v5 = a2[52];
  sub_2F77060(a1, (unsigned int *)v5, *((unsigned int *)a2 + 106));
  v9 = 3LL * *((unsigned int *)a2 + 54);
  v10 = a2[26];
  v70 = v10 + 24LL * *((unsigned int *)a2 + 54);
  while ( v70 != v10 )
  {
    v11 = *(_DWORD *)v10;
    v12 = *(_QWORD *)(v10 + 8);
    v13 = *(_QWORD *)(v10 + 16);
    v14 = *(_DWORD *)v10;
    if ( *(int *)v10 < 0 )
      v14 = *(_DWORD *)(a1 + 320) + (v11 & 0x7FFFFFFF);
    v15 = *(_QWORD *)(a1 + 304);
    v16 = *(unsigned int *)(a1 + 104);
    v17 = *(unsigned __int8 *)(v15 + v14);
    if ( v17 >= (unsigned int)v16 )
      goto LABEL_39;
    v5 = *(_QWORD *)(a1 + 96);
    while ( 1 )
    {
      v15 = v5 + 24LL * v17;
      if ( v14 == *(_DWORD *)v15 )
        break;
      v17 += 256;
      if ( (unsigned int)v16 <= v17 )
        goto LABEL_39;
    }
    if ( v15 == v5 + 24 * v16 )
    {
LABEL_39:
      v19 = 0;
      v18 = 0;
      v83 = 0;
      v89 = 0;
      v80 = 0;
      if ( !(v12 | v13) )
        goto LABEL_10;
    }
    else
    {
      v18 = *(_QWORD *)(v15 + 8);
      v19 = *(_QWORD *)(v15 + 16);
      *(_QWORD *)(v15 + 8) = v18 & ~v12;
      *(_QWORD *)(v15 + 16) = v19 & ~v13;
      v16 = *(_QWORD *)(v10 + 8);
      v15 = *(_QWORD *)(v10 + 16);
      v12 = v16 & ~v18;
      v13 = v15 & ~v19;
      v5 = v19 & ~v15;
      v89 = v18 & ~v16;
      v83 = v5;
      v80 = v5 | v89;
      if ( !(v12 | v13) )
        goto LABEL_10;
    }
    LODWORD(v91) = v11;
    *(_QWORD *)&v92 = v12;
    *((_QWORD *)&v92 + 1) = v13;
    sub_2F77050(a1, v5, v15, v16, v7, v8, v11, v12, v13);
    v40 = 0;
    if ( *(_BYTE *)(a1 + 58) )
    {
      v41 = *(_QWORD *)(a1 + 8);
      v42 = *(__int64 (**)())(*(_QWORD *)v41 + 432LL);
      if ( v42 != sub_2F73F20 )
        v40 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, _QWORD))v42)(
                v41,
                v5,
                v37,
                v38,
                v39,
                0);
    }
    sub_2F74AE0((__int64 *)(a1 + 72), *(_QWORD **)(a1 + 24), v11, 0, 0, v40, v12, v13);
    v19 = v13;
    v18 = v12;
LABEL_10:
    if ( v80 || !*(_BYTE *)(a1 + 58) || !a3 )
      goto LABEL_13;
    v47 = *(unsigned int *)(a3 + 8);
    v48 = *(__m128i **)a3;
    v49 = *(_DWORD *)(a3 + 8);
    v50 = (__m128i *)(*(_QWORD *)a3 + 24 * v47);
    v51 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(24 * v47) >> 3);
    if ( v51 >> 2 )
    {
      v52 = *(__m128i **)a3;
      v53 = &v48[6 * (v51 >> 2)];
      while ( v11 != v52->m128i_i32[0] )
      {
        if ( v11 == v52[1].m128i_i32[2] )
        {
          v52 = (__m128i *)((char *)v52 + 24);
          goto LABEL_65;
        }
        if ( v11 == v52[3].m128i_i32[0] )
        {
          v52 += 3;
          goto LABEL_65;
        }
        if ( v11 == v52[4].m128i_i32[2] )
        {
          v52 = (__m128i *)((char *)v52 + 72);
          goto LABEL_65;
        }
        v52 += 6;
        if ( v53 == v52 )
        {
          v51 = 0xAAAAAAAAAAAAAAABLL * (((char *)v50 - (char *)v52) >> 3);
          goto LABEL_90;
        }
      }
      goto LABEL_65;
    }
    v52 = *(__m128i **)a3;
LABEL_90:
    if ( v51 == 2 )
      goto LABEL_100;
    if ( v51 != 3 )
    {
      if ( v51 != 1 )
        goto LABEL_93;
LABEL_102:
      if ( v11 != v52->m128i_i32[0] )
      {
LABEL_93:
        v60 = *(unsigned int *)(a3 + 12);
        if ( v47 >= v60 )
        {
          v63 = v47 + 1;
          LODWORD(v91) = v11;
          v64 = (const __m128i *)&v91;
          v92 = 0u;
          if ( v60 < v47 + 1 )
          {
            v65 = (const void *)(a3 + 16);
            if ( v48 > (__m128i *)&v91 || v50 <= (__m128i *)&v91 )
            {
              v82 = v18;
              v68 = v19;
              sub_C8D5F0(a3, v65, v63, 0x18u, v63, v8);
              v19 = v68;
              v18 = v82;
              v50 = (__m128i *)(*(_QWORD *)a3 + 24LL * *(unsigned int *)(a3 + 8));
              v64 = (const __m128i *)&v91;
            }
            else
            {
              v66 = v19;
              v67 = v18;
              sub_C8D5F0(a3, v65, v63, 0x18u, v63, v8);
              v18 = v67;
              v19 = v66;
              v64 = (const __m128i *)(*(_QWORD *)a3 + (char *)&v91 - (char *)v48);
              v50 = (__m128i *)(*(_QWORD *)a3 + 24LL * *(unsigned int *)(a3 + 8));
            }
          }
          *v50 = _mm_loadu_si128(v64);
          v50[1].m128i_i64[0] = v64[1].m128i_i64[0];
          ++*(_DWORD *)(a3 + 8);
        }
        else
        {
          if ( v50 )
          {
            v50->m128i_i32[0] = v11;
            v50->m128i_i64[1] = 0;
            v50[1].m128i_i64[0] = 0;
            v49 = *(_DWORD *)(a3 + 8);
          }
          *(_DWORD *)(a3 + 8) = v49 + 1;
        }
        goto LABEL_13;
      }
      goto LABEL_65;
    }
    if ( v11 != v52->m128i_i32[0] )
    {
      v52 = (__m128i *)((char *)v52 + 24);
LABEL_100:
      if ( v11 != v52->m128i_i32[0] )
      {
        v52 = (__m128i *)((char *)v52 + 24);
        goto LABEL_102;
      }
    }
LABEL_65:
    if ( v50 == v52 )
      goto LABEL_93;
    v52->m128i_i64[1] = 0;
    v52[1].m128i_i64[0] = 0;
LABEL_13:
    v5 = v11;
    v10 += 24;
    sub_2F74F40(a1, v11, v18, v19, v89, v83);
  }
  v81 = 0;
  if ( *(_BYTE *)(a1 + 56) )
  {
    v6 = *(_QWORD *)(a1 + 64);
    v55 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL);
    v56 = v6;
    if ( (*(_DWORD *)(v6 + 44) & 4) != 0 )
    {
      do
        v56 = *(_QWORD *)v56 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v56 + 44) & 4) != 0 );
    }
    for ( ; (*(_BYTE *)(v6 + 44) & 8) != 0; v6 = *(_QWORD *)(v6 + 8) )
      ;
    for ( i = *(_QWORD *)(v6 + 8); i != v56; v56 = *(_QWORD *)(v56 + 8) )
    {
      v58 = *(unsigned __int16 *)(v56 + 68);
      v6 = (unsigned int)(v58 - 14);
      if ( (unsigned __int16)(v58 - 14) > 4u && (_WORD)v58 != 24 )
        break;
    }
    v5 = *(_QWORD *)(v55 + 128);
    v59 = *(_DWORD *)(v55 + 144);
    if ( v59 )
    {
      v7 = v59 - 1;
      v6 = (unsigned int)v7 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
      v9 = v5 + 16 * v6;
      v8 = *(_QWORD *)v9;
      if ( v56 == *(_QWORD *)v9 )
      {
LABEL_80:
        v81 = *(_QWORD *)(v9 + 8) & 0xFFFFFFFFFFFFFFF8LL | 4;
        goto LABEL_15;
      }
      v61 = 1;
      while ( v8 != -4096 )
      {
        v62 = v61 + 1;
        v6 = (unsigned int)v7 & (v61 + (_DWORD)v6);
        v9 = v5 + 16LL * (unsigned int)v6;
        v8 = *(_QWORD *)v9;
        if ( *(_QWORD *)v9 == v56 )
          goto LABEL_80;
        v61 = v62;
      }
    }
    v9 = v5 + 16LL * v59;
    goto LABEL_80;
  }
LABEL_15:
  v20 = *a2;
  result = *a2 + 24LL * *((unsigned int *)a2 + 2);
  v90 = result;
  if ( result != *a2 )
  {
    while ( 1 )
    {
      v22 = *(_DWORD *)v20;
      result = sub_2F74C60(a1 + 96, v5, v9, v6, v7, v8, *(_QWORD *)v20, *(_QWORD *)(v20 + 8), *(_QWORD *)(v20 + 16));
      v23 = *(_QWORD *)(v20 + 8);
      v24 = result;
      v6 = v25;
      v9 = *(_QWORD *)(v20 + 16);
      v7 = v23 | result;
      v8 = v9 | v6;
      if ( (v23 | result) == result && v8 == v6 )
        goto LABEL_18;
      if ( v6 | result )
        goto LABEL_57;
      if ( !a3 )
        goto LABEL_56;
      if ( *(_BYTE *)(a1 + 58) )
        break;
      *(_QWORD *)&v92 = *(_QWORD *)(v20 + 8);
      LODWORD(v91) = v22;
      *((_QWORD *)&v92 + 1) = v9;
      v72 = v6;
      v76 = v23 | result;
      v86 = v9 | v6;
      sub_2F747D0(a3, v5, v9, v6, v7, v8, v22, v92, v9);
      v8 = v86;
      v7 = v76;
      v6 = v72;
      if ( *(_BYTE *)(a1 + 56) )
      {
LABEL_68:
        v77 = v7;
        v73 = v6;
        v87 = v8;
        *(_QWORD *)&v54 = sub_2F78130(a1, v22, v81);
        v8 = v87;
        v7 = v77;
        v6 = v73;
        if ( v54 != 0 )
        {
          LODWORD(v91) = v22;
          v92 = v54;
          sub_2F77050(a1, v22, *((__int64 *)&v54 + 1), v73, v77, v87, v22, v54, *((__int64 *)&v54 + 1));
          v6 = v73;
          v7 = v77;
          v8 = v87;
        }
      }
LABEL_57:
      v5 = v22;
      result = sub_2F74DB0(a1, v22, v24, v6, v7, v8);
LABEL_18:
      v20 += 24;
      if ( v90 == v20 )
        goto LABEL_19;
    }
    v43 = *(__m128i **)a3;
    v44 = 24LL * *(unsigned int *)(a3 + 8);
    v84 = (char *)(*(_QWORD *)a3 + v44);
    v45 = 0xAAAAAAAAAAAAAAABLL * (v44 >> 3);
    if ( v45 >> 2 )
    {
      v46 = &v43[6 * (v45 >> 2)];
      while ( v43->m128i_i32[0] != v22 )
      {
        if ( v43[1].m128i_i32[2] == v22 )
        {
          v43 = (__m128i *)((char *)v43 + 24);
          break;
        }
        if ( v43[3].m128i_i32[0] == v22 )
        {
          v43 += 3;
          break;
        }
        if ( v43[4].m128i_i32[2] == v22 )
        {
          v43 = (__m128i *)((char *)v43 + 72);
          break;
        }
        v43 += 6;
        if ( v46 == v43 )
        {
          v45 = 0xAAAAAAAAAAAAAAABLL * ((v84 - (char *)v43) >> 3);
          goto LABEL_82;
        }
      }
LABEL_54:
      if ( v84 != (char *)v43 )
      {
        *(_QWORD *)&v92 = *(_QWORD *)(v20 + 8);
        LODWORD(v91) = v22;
        *((_QWORD *)&v92 + 1) = v9;
        v71 = v6;
        v75 = v7;
        v85 = v9 | v6;
        sub_2F74650(a3, 0xAAAAAAAAAAAAAAABLL, v9, v6, v7, v8, v22, v23, v9);
        v8 = v85;
        v7 = v75;
        v6 = v71;
        goto LABEL_56;
      }
LABEL_86:
      *(_QWORD *)&v92 = *(_QWORD *)(v20 + 8);
      LODWORD(v91) = v22;
      *((_QWORD *)&v92 + 1) = v9;
      v74 = v6;
      v78 = v7;
      v88 = v9 | v6;
      sub_2F747D0(a3, 0xAAAAAAAAAAAAAAABLL, v9, v6, v7, v8, v22, v23, v9);
      v6 = v74;
      v7 = v78;
      v8 = v88;
LABEL_56:
      if ( *(_BYTE *)(a1 + 56) )
        goto LABEL_68;
      goto LABEL_57;
    }
LABEL_82:
    if ( v45 != 2 )
    {
      if ( v45 != 3 )
      {
        if ( v45 != 1 )
          goto LABEL_86;
        goto LABEL_85;
      }
      if ( v43->m128i_i32[0] == v22 )
        goto LABEL_54;
      v43 = (__m128i *)((char *)v43 + 24);
    }
    if ( v43->m128i_i32[0] == v22 )
      goto LABEL_54;
    v43 = (__m128i *)((char *)v43 + 24);
LABEL_85:
    if ( v43->m128i_i32[0] != v22 )
      goto LABEL_86;
    goto LABEL_54;
  }
LABEL_19:
  if ( *(_BYTE *)(a1 + 57) )
  {
    v26 = a2[26];
    result = 3LL * *((unsigned int *)a2 + 54);
    for ( j = v26 + 24LL * *((unsigned int *)a2 + 54); j != v26; v26 += 24 )
    {
      v28 = *(_DWORD *)v26;
      if ( *(int *)v26 < 0 )
      {
        v29 = v28 & 0x7FFFFFFF;
        v30 = *(unsigned int *)(a1 + 104);
        v31 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 304) + (v28 & 0x7FFFFFFFu) + *(_DWORD *)(a1 + 320));
        if ( v31 >= (unsigned int)v30 )
          goto LABEL_30;
        v7 = *(_QWORD *)(a1 + 96);
        while ( 1 )
        {
          v32 = v7 + 24LL * v31;
          if ( (v28 & 0x7FFFFFFF) + *(_DWORD *)(a1 + 320) == *(_DWORD *)v32 )
            break;
          v31 += 256;
          if ( (unsigned int)v30 <= v31 )
            goto LABEL_30;
        }
        if ( v32 == v7 + 24 * v30
          || (result = *(_QWORD *)(v32 + 16) & *(_QWORD *)(v26 + 16), (*(_OWORD *)(v32 + 8) & *(_OWORD *)(v26 + 8)) == 0) )
        {
LABEL_30:
          v33 = (_BYTE *)(*(_QWORD *)(a1 + 376) + v29);
          v34 = *(_DWORD *)(a1 + 336);
          v35 = (unsigned __int8)*v33;
          if ( v35 >= v34 )
            goto LABEL_35;
          v7 = *(_QWORD *)(a1 + 328);
          while ( 1 )
          {
            v36 = (_DWORD *)(v7 + 4LL * v35);
            if ( v29 == (*v36 & 0x7FFFFFFF) )
              break;
            v35 += 256;
            if ( v34 <= v35 )
              goto LABEL_35;
          }
          result = v7 + 4LL * v34;
          if ( v36 == (_DWORD *)result )
          {
LABEL_35:
            *v33 = v34;
            result = *(unsigned int *)(a1 + 336);
            if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 340) )
            {
              sub_C8D5F0(a1 + 328, (const void *)(a1 + 344), result + 1, 4u, v7, (__int64)v33);
              result = *(unsigned int *)(a1 + 336);
            }
            *(_DWORD *)(*(_QWORD *)(a1 + 328) + 4 * result) = v28;
            ++*(_DWORD *)(a1 + 336);
          }
        }
      }
    }
  }
  return result;
}
