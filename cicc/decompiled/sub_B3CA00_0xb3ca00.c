// Function: sub_B3CA00
// Address: 0xb3ca00
//
__int64 __fastcall sub_B3CA00(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  _BYTE *v4; // r10
  __int64 v5; // r15
  __int64 v7; // r13
  char *v8; // r12
  _BYTE *v9; // rax
  int v10; // ebx
  int v11; // edx
  unsigned int v12; // eax
  char v13; // al
  char v14; // al
  unsigned int v15; // r10d
  _BYTE *v17; // rdx
  unsigned __int8 v18; // al
  __int64 v19; // rax
  int v20; // esi
  size_t v21; // rcx
  __m128i *v22; // rdi
  __m128i *v23; // rax
  __int64 v24; // rsi
  _BYTE *v25; // rax
  __int64 v26; // rdx
  _BYTE *v27; // rax
  __int64 v28; // rsi
  int v29; // edi
  size_t v30; // rcx
  __m128i *v31; // rsi
  __m128i *v32; // rdx
  __int64 v33; // rdi
  unsigned __int64 v34; // rcx
  __int64 v35; // rsi
  __int64 v36; // rbx
  __int64 v37; // rdx
  __int64 v38; // rax
  unsigned __int8 *v39; // rcx
  size_t v40; // r11
  __int64 v41; // rax
  __int64 v42; // rsi
  int v43; // edi
  size_t v44; // rdx
  __m128i *v45; // rsi
  __m128i *v46; // rax
  __int64 v47; // rdi
  unsigned int v48; // eax
  unsigned __int64 v49; // rsi
  __int64 v50; // rax
  _DWORD *v51; // rax
  __int64 v52; // rax
  _QWORD *v53; // rdi
  int v54; // ecx
  char *v55; // r11
  unsigned __int64 v56; // rcx
  _QWORD *v57; // rax
  __int64 v58; // rax
  int v59; // esi
  size_t v60; // rdx
  __m128i *v61; // rcx
  __m128i *v62; // rax
  __int64 v63; // rsi
  __int64 v64; // rdi
  __int16 v65; // ax
  __int64 v66; // rax
  __int64 v67; // rsi
  int v68; // ecx
  size_t v69; // rdx
  __m128i *v70; // rsi
  __m128i *v71; // rax
  __int64 v72; // rcx
  size_t v73; // r12
  _QWORD *v74; // rbx
  __int64 v75; // rax
  _QWORD *v76; // r15
  size_t v77; // r12
  size_t v78; // r12
  __int64 v79; // rsi
  size_t v80; // r12
  __int64 v81; // rsi
  __int64 v82; // [rsp+8h] [rbp-88h]
  int n; // [rsp+18h] [rbp-78h]
  size_t nb; // [rsp+18h] [rbp-78h]
  size_t nc; // [rsp+18h] [rbp-78h]
  size_t na; // [rsp+18h] [rbp-78h]
  _BYTE *v88; // [rsp+20h] [rbp-70h]
  _BYTE *v89; // [rsp+20h] [rbp-70h]
  _BYTE *v90; // [rsp+20h] [rbp-70h]
  unsigned __int8 *v91; // [rsp+20h] [rbp-70h]
  unsigned __int8 *v92; // [rsp+20h] [rbp-70h]
  unsigned __int8 *v93; // [rsp+20h] [rbp-70h]
  unsigned __int8 *v94; // [rsp+20h] [rbp-70h]
  char *v95; // [rsp+20h] [rbp-70h]
  __int64 v96; // [rsp+20h] [rbp-70h]
  unsigned __int8 *v97; // [rsp+20h] [rbp-70h]
  _BYTE *v98; // [rsp+20h] [rbp-70h]
  unsigned int v99; // [rsp+28h] [rbp-68h]
  _BYTE *v100; // [rsp+28h] [rbp-68h]
  _BYTE *v101; // [rsp+28h] [rbp-68h]
  _BYTE *v102; // [rsp+28h] [rbp-68h]
  unsigned __int64 v103; // [rsp+38h] [rbp-58h] BYREF
  _QWORD *v104; // [rsp+40h] [rbp-50h] BYREF
  __int64 v105; // [rsp+48h] [rbp-48h]
  _QWORD v106[8]; // [rsp+50h] [rbp-40h] BYREF

  v4 = a2;
  v5 = a1 + 16;
  v7 = (__int64)&a2[a3];
  v8 = a2;
  if ( !a3 )
  {
    *(_BYTE *)(a1 + 11) = 0;
    goto LABEL_5;
  }
  v9 = a2;
  v10 = 0;
  do
  {
    v11 = *v9++ == 124;
    v10 += v11;
  }
  while ( (_BYTE *)v7 != v9 );
  v12 = v10 + 1;
  n = v10 + 1;
  *(_BYTE *)(a1 + 11) = (unsigned int)(v10 + 1) > 1;
  if ( (unsigned int)(v10 + 1) > 1 )
  {
    v34 = *(unsigned int *)(a1 + 72);
    v35 = v12;
    if ( v12 == v34 )
    {
      v37 = *(_QWORD *)(a1 + 64);
      goto LABEL_68;
    }
    v36 = 56LL * v12;
    if ( v12 < v34 )
    {
      v37 = *(_QWORD *)(a1 + 64);
      v96 = v37 + 56 * v34;
      v82 = v37 + v36;
      if ( v96 == v37 + v36 )
        goto LABEL_67;
      do
      {
        v96 -= 56;
        v74 = *(_QWORD **)(v96 + 8);
        v75 = 4LL * *(unsigned int *)(v96 + 16);
        v76 = &v74[v75];
        if ( v74 != &v74[v75] )
        {
          do
          {
            v76 -= 4;
            if ( (_QWORD *)*v76 != v76 + 2 )
            {
              v101 = v4;
              v35 = v76[2] + 1LL;
              j_j___libc_free_0(*v76, v35);
              v4 = v101;
            }
          }
          while ( v74 != v76 );
          v74 = *(_QWORD **)(v96 + 8);
        }
        if ( v74 != (_QWORD *)(v96 + 24) )
        {
          v102 = v4;
          _libc_free(v74, v35);
          v4 = v102;
        }
      }
      while ( v82 != v96 );
    }
    else
    {
      if ( v12 > (unsigned __int64)*(unsigned int *)(a1 + 76) )
      {
        v100 = v4;
        sub_B3C890(a1 + 64, v12);
        v34 = *(unsigned int *)(a1 + 72);
        v4 = v100;
      }
      v37 = *(_QWORD *)(a1 + 64);
      v38 = v37 + 56 * v34;
      if ( v38 == v37 + v36 )
        goto LABEL_67;
      do
      {
        if ( v38 )
        {
          *(_QWORD *)(v38 + 48) = 0;
          *(_OWORD *)v38 = 0;
          *(_OWORD *)(v38 + 16) = 0;
          *(_DWORD *)v38 = -1;
          *(_QWORD *)(v38 + 8) = v38 + 24;
          *(_DWORD *)(v38 + 20) = 1;
          *(_OWORD *)(v38 + 32) = 0;
        }
        v38 += 56;
      }
      while ( v37 + v36 != v38 );
    }
    v37 = *(_QWORD *)(a1 + 64);
LABEL_67:
    *(_DWORD *)(a1 + 72) = n;
LABEL_68:
    v5 = v37 + 8;
  }
LABEL_5:
  *(_BYTE *)(a1 + 10) = 0;
  *(_QWORD *)a1 = 0xFFFFFFFF00000000LL;
  *(_WORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 12) = 0;
  v13 = *v4;
  if ( *v4 != 126 )
  {
    switch ( v13 )
    {
      case '=':
        *(_DWORD *)a1 = 1;
        v13 = v4[1];
        v8 = v4 + 1;
        break;
      case '!':
        *(_DWORD *)a1 = 3;
        v8 = v4 + 1;
        if ( v4[1] == 42 )
          goto LABEL_9;
LABEL_12:
        if ( v8 != (char *)v7 )
          goto LABEL_13;
        return 1;
      case '*':
        goto LABEL_9;
    }
    goto LABEL_12;
  }
  v8 = v4 + 1;
  *(_DWORD *)a1 = 2;
  if ( v4 + 1 != (_BYTE *)v7 )
  {
    v14 = v4[1];
    if ( v14 == 123 )
      goto LABEL_14;
    return 1;
  }
  if ( v4[1] != 42 )
    return 1;
  v8 = (char *)v7;
LABEL_9:
  *(_BYTE *)(a1 + 10) = 1;
  while ( 1 )
  {
    while ( 1 )
    {
      if ( ++v8 == (char *)v7 )
        return 1;
LABEL_13:
      v14 = *v8;
LABEL_14:
      if ( v14 != 38 )
        break;
      if ( *(_DWORD *)a1 != 1 || *(_BYTE *)(a1 + 8) )
        return 1;
      *(_BYTE *)(a1 + 8) = 1;
    }
    if ( v14 > 38 )
      break;
    if ( v14 == 35 )
      return 1;
    if ( v14 != 37 )
      goto LABEL_25;
    if ( *(_DWORD *)a1 == 2 || *(_BYTE *)(a1 + 9) )
      return 1;
    *(_BYTE *)(a1 + 9) = 1;
  }
  v15 = 1;
  if ( v14 != 42 )
  {
LABEL_25:
    if ( (char *)v7 == v8 )
      return 0;
    v99 = 0;
    if ( v14 == 123 )
      goto LABEL_42;
LABEL_27:
    if ( (unsigned int)(unsigned __int8)v14 - 48 > 9 )
    {
      if ( v14 == 124 )
      {
        ++v99;
        ++v8;
        v5 = *(_QWORD *)(a1 + 64) + 56LL * v99 + 8;
        goto LABEL_40;
      }
      if ( v14 != 94 )
      {
        if ( v14 != 64 )
        {
          v17 = v8 + 1;
          v104 = v106;
          v18 = *v8;
          v105 = 1;
          LOWORD(v106[0]) = v18;
          v19 = *(unsigned int *)(v5 + 8);
          v20 = *(_DWORD *)(v5 + 8);
          if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(v5 + 12) )
          {
            v78 = *(_QWORD *)v5;
            v79 = v19 + 1;
            v98 = v17;
            if ( *(_QWORD *)v5 > (unsigned __int64)&v104 || (unsigned __int64)&v104 >= v78 + 32 * v19 )
            {
              sub_95D880(v5, v79);
              v19 = *(unsigned int *)(v5 + 8);
              v21 = *(_QWORD *)v5;
              v22 = (__m128i *)&v104;
              v17 = v98;
              v20 = *(_DWORD *)(v5 + 8);
            }
            else
            {
              sub_95D880(v5, v79);
              v21 = *(_QWORD *)v5;
              v19 = *(unsigned int *)(v5 + 8);
              v17 = v98;
              v22 = (__m128i *)((char *)&v104 + *(_QWORD *)v5 - v78);
              v20 = *(_DWORD *)(v5 + 8);
            }
          }
          else
          {
            v21 = *(_QWORD *)v5;
            v22 = (__m128i *)&v104;
          }
          v23 = (__m128i *)(v21 + 32 * v19);
          if ( v23 )
          {
            v23->m128i_i64[0] = (__int64)v23[1].m128i_i64;
            if ( (__m128i *)v22->m128i_i64[0] == &v22[1] )
            {
              v23[1] = _mm_loadu_si128(v22 + 1);
            }
            else
            {
              v23->m128i_i64[0] = v22->m128i_i64[0];
              v23[1].m128i_i64[0] = v22[1].m128i_i64[0];
            }
            v24 = v22->m128i_i64[1];
            v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
            v22->m128i_i64[1] = 0;
            v23->m128i_i64[1] = v24;
            v22[1].m128i_i8[0] = 0;
            v20 = *(_DWORD *)(v5 + 8);
          }
          *(_DWORD *)(v5 + 8) = v20 + 1;
          if ( v104 != v106 )
          {
            v88 = v17;
            j_j___libc_free_0(v104, v106[0] + 1LL);
            v17 = v88;
          }
          v8 = v17;
          goto LABEL_40;
        }
        v54 = (unsigned __int8)v8[1];
        v55 = v8 + 2;
        v104 = v106;
        v56 = v54 - 48;
        v103 = v56;
        v95 = &v8[v56 + 2];
        if ( v56 > 0xF )
        {
          nc = v56;
          v57 = (_QWORD *)sub_22409D0(&v104, &v103, 0);
          v56 = nc;
          v104 = v57;
          v55 = v8 + 2;
          v106[0] = v103;
        }
        else
        {
          if ( v56 == 1 )
          {
            LOBYTE(v106[0]) = v8[2];
            v57 = v106;
LABEL_97:
            v105 = v56;
            *((_BYTE *)v57 + v56) = 0;
            v58 = *(unsigned int *)(v5 + 8);
            v59 = *(_DWORD *)(v5 + 8);
            if ( v58 + 1 > (unsigned __int64)*(unsigned int *)(v5 + 12) )
            {
              v80 = *(_QWORD *)v5;
              v81 = v58 + 1;
              if ( *(_QWORD *)v5 > (unsigned __int64)&v104 || (unsigned __int64)&v104 >= v80 + 32 * v58 )
              {
                sub_95D880(v5, v81);
                v58 = *(unsigned int *)(v5 + 8);
                v60 = *(_QWORD *)v5;
                v61 = (__m128i *)&v104;
                v59 = *(_DWORD *)(v5 + 8);
              }
              else
              {
                sub_95D880(v5, v81);
                v60 = *(_QWORD *)v5;
                v58 = *(unsigned int *)(v5 + 8);
                v61 = (__m128i *)((char *)&v104 + *(_QWORD *)v5 - v80);
                v59 = *(_DWORD *)(v5 + 8);
              }
            }
            else
            {
              v60 = *(_QWORD *)v5;
              v61 = (__m128i *)&v104;
            }
            v62 = (__m128i *)(v60 + 32 * v58);
            if ( v62 )
            {
              v62->m128i_i64[0] = (__int64)v62[1].m128i_i64;
              if ( (__m128i *)v61->m128i_i64[0] == &v61[1] )
              {
                v62[1] = _mm_loadu_si128(v61 + 1);
              }
              else
              {
                v62->m128i_i64[0] = v61->m128i_i64[0];
                v62[1].m128i_i64[0] = v61[1].m128i_i64[0];
              }
              v63 = v61->m128i_i64[1];
              v61->m128i_i64[0] = (__int64)v61[1].m128i_i64;
              v61->m128i_i64[1] = 0;
              v62->m128i_i64[1] = v63;
              v61[1].m128i_i8[0] = 0;
              v59 = *(_DWORD *)(v5 + 8);
            }
            *(_DWORD *)(v5 + 8) = v59 + 1;
            if ( v104 != v106 )
              j_j___libc_free_0(v104, v106[0] + 1LL);
            v8 = v95;
            goto LABEL_40;
          }
          v57 = v106;
          if ( !v56 )
            goto LABEL_97;
        }
        if ( v56 >= 8 )
        {
          *v57 = *(_QWORD *)v55;
          *(_QWORD *)((char *)v57 + v56 - 8) = *(_QWORD *)&v55[v56 - 8];
          qmemcpy(
            (void *)((unsigned __int64)(v57 + 1) & 0xFFFFFFFFFFFFFFF8LL),
            (const void *)(v55 - ((char *)v57 - ((unsigned __int64)(v57 + 1) & 0xFFFFFFFFFFFFFFF8LL))),
            8 * (((unsigned __int64)v57 + v56 - ((unsigned __int64)(v57 + 1) & 0xFFFFFFFFFFFFFFF8LL)) >> 3));
        }
        else if ( (v56 & 4) != 0 )
        {
          *(_DWORD *)v57 = *(_DWORD *)v55;
          *(_DWORD *)((char *)v57 + v56 - 4) = *(_DWORD *)&v55[v56 - 4];
        }
        else
        {
          *(_BYTE *)v57 = *v55;
          if ( (v56 & 2) != 0 )
            *(_WORD *)((char *)v57 + v56 - 2) = *(_WORD *)&v55[v56 - 2];
        }
        v56 = v103;
        v57 = v104;
        goto LABEL_97;
      }
      v104 = v106;
      v65 = *(_WORD *)(v8 + 1);
      v105 = 2;
      LOWORD(v106[0]) = v65;
      BYTE2(v106[0]) = 0;
      v66 = *(unsigned int *)(v5 + 8);
      v67 = v66 + 1;
      v68 = *(_DWORD *)(v5 + 8);
      if ( v66 + 1 > (unsigned __int64)*(unsigned int *)(v5 + 12) )
      {
        if ( *(_QWORD *)v5 > (unsigned __int64)&v104
          || (na = *(_QWORD *)v5, (unsigned __int64)&v104 >= *(_QWORD *)v5 + 32 * v66) )
        {
          sub_95D880(v5, v67);
          v66 = *(unsigned int *)(v5 + 8);
          v69 = *(_QWORD *)v5;
          v70 = (__m128i *)&v104;
          v68 = *(_DWORD *)(v5 + 8);
        }
        else
        {
          sub_95D880(v5, v67);
          v69 = *(_QWORD *)v5;
          v66 = *(unsigned int *)(v5 + 8);
          v70 = (__m128i *)((char *)&v104 + *(_QWORD *)v5 - na);
          v68 = *(_DWORD *)(v5 + 8);
        }
      }
      else
      {
        v69 = *(_QWORD *)v5;
        v70 = (__m128i *)&v104;
      }
      v71 = (__m128i *)(v69 + 32 * v66);
      if ( v71 )
      {
        v71->m128i_i64[0] = (__int64)v71[1].m128i_i64;
        if ( (__m128i *)v70->m128i_i64[0] == &v70[1] )
        {
          v71[1] = _mm_loadu_si128(v70 + 1);
        }
        else
        {
          v71->m128i_i64[0] = v70->m128i_i64[0];
          v71[1].m128i_i64[0] = v70[1].m128i_i64[0];
        }
        v72 = v70->m128i_i64[1];
        v70->m128i_i64[0] = (__int64)v70[1].m128i_i64;
        v70->m128i_i64[1] = 0;
        v71->m128i_i64[1] = v72;
        v70[1].m128i_i8[0] = 0;
        v68 = *(_DWORD *)(v5 + 8);
      }
      *(_DWORD *)(v5 + 8) = v68 + 1;
      if ( v104 != v106 )
        j_j___libc_free_0(v104, v106[0] + 1LL);
      v8 += 3;
LABEL_40:
      while ( (char *)v7 != v8 )
      {
        v14 = *v8;
        if ( *v8 != 123 )
          goto LABEL_27;
LABEL_42:
        LOBYTE(v104) = 125;
        v25 = sub_B3AF10(v8 + 1, v7, (char *)&v104);
        if ( v25 == (_BYTE *)v7 )
          return 1;
        v104 = v106;
        v89 = v25 + 1;
        sub_B3B150((__int64 *)&v104, v8, (__int64)(v25 + 1));
        v26 = *(unsigned int *)(v5 + 8);
        v27 = v89;
        v28 = v26 + 1;
        v29 = *(_DWORD *)(v5 + 8);
        if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(v5 + 12) )
        {
          v73 = *(_QWORD *)v5;
          if ( *(_QWORD *)v5 > (unsigned __int64)&v104 || (unsigned __int64)&v104 >= v73 + 32 * v26 )
          {
            sub_95D880(v5, v28);
            v26 = *(unsigned int *)(v5 + 8);
            v30 = *(_QWORD *)v5;
            v31 = (__m128i *)&v104;
            v27 = v89;
            v29 = *(_DWORD *)(v5 + 8);
          }
          else
          {
            sub_95D880(v5, v28);
            v30 = *(_QWORD *)v5;
            v26 = *(unsigned int *)(v5 + 8);
            v27 = v89;
            v31 = (__m128i *)((char *)&v104 + *(_QWORD *)v5 - v73);
            v29 = *(_DWORD *)(v5 + 8);
          }
        }
        else
        {
          v30 = *(_QWORD *)v5;
          v31 = (__m128i *)&v104;
        }
        v32 = (__m128i *)(v30 + 32 * v26);
        if ( v32 )
        {
          v32->m128i_i64[0] = (__int64)v32[1].m128i_i64;
          if ( (__m128i *)v31->m128i_i64[0] == &v31[1] )
          {
            v32[1] = _mm_loadu_si128(v31 + 1);
          }
          else
          {
            v32->m128i_i64[0] = v31->m128i_i64[0];
            v32[1].m128i_i64[0] = v31[1].m128i_i64[0];
          }
          v33 = v31->m128i_i64[1];
          v31->m128i_i64[0] = (__int64)v31[1].m128i_i64;
          v31->m128i_i64[1] = 0;
          v32->m128i_i64[1] = v33;
          v31[1].m128i_i8[0] = 0;
          v29 = *(_DWORD *)(v5 + 8);
        }
        *(_DWORD *)(v5 + 8) = v29 + 1;
        if ( v104 != v106 )
        {
          v90 = v27;
          j_j___libc_free_0(v104, v106[0] + 1LL);
          v27 = v90;
        }
        v8 = v27;
      }
      return 0;
    }
    if ( v8 == (char *)v7 )
    {
      v103 = 0;
      v39 = (unsigned __int8 *)v7;
      v104 = v106;
    }
    else
    {
      v39 = (unsigned __int8 *)v8;
      do
        ++v39;
      while ( v39 != (unsigned __int8 *)v7 && (unsigned int)*v39 - 48 <= 9 );
      v40 = v39 - (unsigned __int8 *)v8;
      v104 = v106;
      v103 = v39 - (unsigned __int8 *)v8;
      if ( (unsigned __int64)(v39 - (unsigned __int8 *)v8) > 0xF )
      {
        nb = v39 - (unsigned __int8 *)v8;
        v93 = v39;
        v52 = sub_22409D0(&v104, &v103, 0);
        v39 = v93;
        v40 = nb;
        v104 = (_QWORD *)v52;
        v53 = (_QWORD *)v52;
        v106[0] = v103;
LABEL_93:
        v94 = v39;
        memcpy(v53, v8, v40);
        v39 = v94;
        goto LABEL_76;
      }
      if ( v40 == 1 )
      {
        LOBYTE(v106[0]) = *v8;
        goto LABEL_76;
      }
      if ( v40 )
      {
        v53 = v106;
        goto LABEL_93;
      }
    }
LABEL_76:
    v105 = v103;
    *((_BYTE *)v104 + v103) = 0;
    v41 = *(unsigned int *)(v5 + 8);
    v42 = v41 + 1;
    v43 = *(_DWORD *)(v5 + 8);
    if ( v41 + 1 > (unsigned __int64)*(unsigned int *)(v5 + 12) )
    {
      v77 = *(_QWORD *)v5;
      v97 = v39;
      if ( *(_QWORD *)v5 > (unsigned __int64)&v104 || (unsigned __int64)&v104 >= v77 + 32 * v41 )
      {
        sub_95D880(v5, v42);
        v41 = *(unsigned int *)(v5 + 8);
        v44 = *(_QWORD *)v5;
        v45 = (__m128i *)&v104;
        v39 = v97;
        v43 = *(_DWORD *)(v5 + 8);
      }
      else
      {
        sub_95D880(v5, v42);
        v44 = *(_QWORD *)v5;
        v41 = *(unsigned int *)(v5 + 8);
        v39 = v97;
        v45 = (__m128i *)((char *)&v104 + *(_QWORD *)v5 - v77);
        v43 = *(_DWORD *)(v5 + 8);
      }
    }
    else
    {
      v44 = *(_QWORD *)v5;
      v45 = (__m128i *)&v104;
    }
    v46 = (__m128i *)(v44 + 32 * v41);
    if ( v46 )
    {
      v46->m128i_i64[0] = (__int64)v46[1].m128i_i64;
      if ( (__m128i *)v45->m128i_i64[0] == &v45[1] )
      {
        v46[1] = _mm_loadu_si128(v45 + 1);
      }
      else
      {
        v46->m128i_i64[0] = v45->m128i_i64[0];
        v46[1].m128i_i64[0] = v45[1].m128i_i64[0];
      }
      v47 = v45->m128i_i64[1];
      v45->m128i_i64[0] = (__int64)v45[1].m128i_i64;
      v45->m128i_i64[1] = 0;
      v46->m128i_i64[1] = v47;
      v45[1].m128i_i8[0] = 0;
      v43 = *(_DWORD *)(v5 + 8);
    }
    *(_DWORD *)(v5 + 8) = v43 + 1;
    if ( v104 != v106 )
    {
      v91 = v39;
      j_j___libc_free_0(v104, v106[0] + 1LL);
      v39 = v91;
    }
    v92 = v39;
    v48 = strtol(*(const char **)(*(_QWORD *)v5 + 32LL * *(unsigned int *)(v5 + 8) - 32), 0, 10);
    v49 = *(unsigned int *)(a4 + 8);
    if ( v48 < v49 )
    {
      v50 = *(_QWORD *)a4 + 192LL * v48;
      if ( *(_DWORD *)v50 == 1 && !*(_DWORD *)a1 )
      {
        if ( *(_BYTE *)(a1 + 11) )
        {
          if ( *(_DWORD *)(v50 + 72) > v99 )
          {
            v51 = (_DWORD *)(*(_QWORD *)(v50 + 64) + 56LL * v99);
            if ( *v51 == -1 )
            {
              *v51 = *(_DWORD *)(a4 + 8);
              v8 = (char *)v92;
              goto LABEL_40;
            }
          }
        }
        else
        {
          v64 = *(int *)(v50 + 4);
          if ( (_DWORD)v64 == -1 || v64 == v49 )
          {
            *(_DWORD *)(v50 + 4) = v49;
            v8 = (char *)v92;
            goto LABEL_40;
          }
        }
      }
    }
    return 1;
  }
  return v15;
}
