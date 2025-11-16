// Function: sub_15EBA30
// Address: 0x15eba30
//
__int64 __fastcall sub_15EBA30(__int64 a1, _BYTE *a2, __int64 a3, unsigned __int64 a4, __int64 a5, size_t a6)
{
  __int64 v6; // r14
  _BYTE *v8; // r12
  __int64 v9; // rbx
  _BYTE *v10; // rax
  int v11; // edx
  char v12; // al
  char v13; // al
  unsigned __int8 v15; // al
  __int64 v16; // rdx
  __m128i *v17; // rax
  _BYTE *v18; // rax
  __int64 *v19; // r9
  __int64 v20; // rcx
  _BYTE *v21; // rdx
  __m128i *v22; // rax
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // r15
  __int64 v26; // rax
  _QWORD *v27; // r14
  __int8 v28; // al
  __int64 v29; // rdx
  __int64 v30; // rcx
  __m128i *v31; // rax
  unsigned int v32; // eax
  __int64 v33; // rax
  _DWORD *v34; // rax
  __m128i *v35; // rdi
  __m128i *v36; // rdi
  __int16 v37; // ax
  __int64 v38; // rdx
  __m128i *v39; // rax
  __m128i *v40; // rdi
  __int64 v41; // rsi
  __m128i *v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rax
  __m128i *v45; // rdi
  unsigned __int64 v46; // [rsp+8h] [rbp-88h]
  int v47; // [rsp+14h] [rbp-7Ch]
  size_t n; // [rsp+18h] [rbp-78h]
  size_t na; // [rsp+18h] [rbp-78h]
  __int64 v50; // [rsp+20h] [rbp-70h]
  _BYTE *v51; // [rsp+20h] [rbp-70h]
  unsigned __int64 v52; // [rsp+20h] [rbp-70h]
  _BYTE *v53; // [rsp+20h] [rbp-70h]
  unsigned int v54; // [rsp+28h] [rbp-68h]
  unsigned __int64 v55; // [rsp+28h] [rbp-68h]
  unsigned __int64 v56; // [rsp+28h] [rbp-68h]
  __int64 v57; // [rsp+38h] [rbp-58h] BYREF
  __m128i *v58; // [rsp+40h] [rbp-50h] BYREF
  __int64 v59; // [rsp+48h] [rbp-48h]
  __m128i v60; // [rsp+50h] [rbp-40h] BYREF

  v6 = a1 + 16;
  v8 = a2;
  v9 = (__int64)&a2[a3];
  v46 = a4;
  if ( !a3 )
  {
    *(_BYTE *)(a1 + 11) = 0;
    goto LABEL_5;
  }
  v10 = a2;
  v11 = 0;
  do
  {
    a4 = *v10++ == 124;
    v11 += a4;
  }
  while ( (_BYTE *)v9 != v10 );
  v47 = v11 + 1;
  *(_BYTE *)(a1 + 11) = (unsigned int)(v11 + 1) > 1;
  if ( (unsigned int)(v11 + 1) > 1 )
  {
    v23 = (unsigned int)(v11 + 1);
    v24 = *(unsigned int *)(a1 + 72);
    if ( v23 >= v24 )
    {
      if ( v23 <= v24 )
      {
        a5 = *(_QWORD *)(a1 + 64);
        goto LABEL_92;
      }
      if ( v23 > *(unsigned int *)(a1 + 76) )
      {
        v56 = v23;
        sub_15EB820(a1 + 64, v23);
        v24 = *(unsigned int *)(a1 + 72);
        v23 = v56;
      }
      a5 = *(_QWORD *)(a1 + 64);
      v43 = a5 + 56 * v24;
      a4 = a5 + 56 * v23;
      if ( v43 != a4 )
      {
        do
        {
          if ( v43 )
          {
            *(_QWORD *)(v43 + 48) = 0;
            *(_OWORD *)v43 = 0;
            *(_OWORD *)(v43 + 16) = 0;
            *(_DWORD *)v43 = -1;
            *(_QWORD *)(v43 + 8) = v43 + 24;
            *(_DWORD *)(v43 + 20) = 1;
            *(_OWORD *)(v43 + 32) = 0;
          }
          v43 += 56;
        }
        while ( a4 != v43 );
        goto LABEL_65;
      }
    }
    else
    {
      a5 = *(_QWORD *)(a1 + 64);
      a4 = a5 + 56 * v24;
      v55 = a4;
      n = a5 + 56 * v23;
      if ( a4 != n )
      {
        do
        {
          v55 -= 56LL;
          v25 = *(_QWORD *)(v55 + 8);
          v26 = 32LL * *(unsigned int *)(v55 + 16);
          v27 = (_QWORD *)(v25 + v26);
          if ( v25 != v25 + v26 )
          {
            do
            {
              v27 -= 4;
              if ( (_QWORD *)*v27 != v27 + 2 )
                j_j___libc_free_0(*v27, v27[2] + 1LL);
            }
            while ( (_QWORD *)v25 != v27 );
            v25 = *(_QWORD *)(v55 + 8);
          }
          if ( v25 != v55 + 24 )
            _libc_free(v25);
        }
        while ( n != v55 );
LABEL_65:
        a5 = *(_QWORD *)(a1 + 64);
      }
    }
    *(_DWORD *)(a1 + 72) = v47;
LABEL_92:
    v6 = a5 + 8;
  }
LABEL_5:
  *(_BYTE *)(a1 + 10) = 0;
  *(_QWORD *)a1 = 0xFFFFFFFF00000000LL;
  *(_WORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 12) = 0;
  v12 = *a2;
  if ( *a2 == 126 )
  {
    *(_DWORD *)a1 = 2;
    v8 = a2 + 1;
    if ( a2 + 1 != (_BYTE *)v9 )
    {
      v13 = a2[1];
      if ( v13 == 123 )
        goto LABEL_10;
      goto LABEL_18;
    }
    if ( a2[1] != 42 )
    {
LABEL_18:
      LODWORD(a6) = 1;
      return (unsigned int)a6;
    }
    v8 = (_BYTE *)v9;
    goto LABEL_39;
  }
  if ( v12 != 61 )
  {
    if ( v12 != 42 )
      goto LABEL_8;
LABEL_39:
    *(_BYTE *)(a1 + 10) = 1;
    goto LABEL_17;
  }
  *(_DWORD *)a1 = 1;
  v8 = a2 + 1;
  if ( a2[1] == 42 )
    goto LABEL_39;
LABEL_8:
  if ( v8 == (_BYTE *)v9 )
    goto LABEL_18;
  while ( 1 )
  {
    v13 = *v8;
LABEL_10:
    if ( v13 == 38 )
    {
      if ( *(_DWORD *)a1 != 1 || *(_BYTE *)(a1 + 8) )
        goto LABEL_18;
      *(_BYTE *)(a1 + 8) = 1;
      goto LABEL_17;
    }
    if ( v13 > 38 )
      break;
    if ( v13 == 35 )
      goto LABEL_18;
    if ( v13 != 37 )
      goto LABEL_21;
    if ( *(_DWORD *)a1 == 2 || *(_BYTE *)(a1 + 9) )
      goto LABEL_18;
    *(_BYTE *)(a1 + 9) = 1;
LABEL_17:
    if ( ++v8 == (_BYTE *)v9 )
      goto LABEL_18;
  }
  a6 = 1;
  if ( v13 != 42 )
  {
LABEL_21:
    if ( (_BYTE *)v9 != v8 )
    {
      v54 = 0;
      while ( 1 )
      {
        if ( v13 == 123 )
        {
          LOBYTE(v58) = 125;
          v18 = sub_15EA350(v8 + 1, v9, (char *)&v58);
          if ( (_BYTE *)v9 == v18 )
            goto LABEL_18;
          v58 = &v60;
          v50 = (__int64)(v18 + 1);
          sub_15EA2A0(v19, v8, (__int64)(v18 + 1));
          v20 = *(unsigned int *)(v6 + 8);
          v21 = (_BYTE *)v50;
          if ( (unsigned int)v20 >= *(_DWORD *)(v6 + 12) )
          {
            sub_12BE710(v6, 0, v50, v20, a5, a6);
            LODWORD(v20) = *(_DWORD *)(v6 + 8);
            v21 = (_BYTE *)v50;
          }
          v22 = (__m128i *)(*(_QWORD *)v6 + 32LL * (unsigned int)v20);
          if ( v22 )
          {
            v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
            if ( v58 == &v60 )
            {
              v22[1] = _mm_load_si128(&v60);
            }
            else
            {
              v22->m128i_i64[0] = (__int64)v58;
              v22[1].m128i_i64[0] = v60.m128i_i64[0];
            }
            a4 = v59;
            v22->m128i_i64[1] = v59;
            v59 = 0;
            v60.m128i_i8[0] = 0;
            ++*(_DWORD *)(v6 + 8);
          }
          else
          {
            v35 = v58;
            a4 = (unsigned int)(v20 + 1);
            *(_DWORD *)(v6 + 8) = a4;
            if ( v35 != &v60 )
            {
              v51 = v21;
              j_j___libc_free_0(v35, v60.m128i_i64[0] + 1);
              v21 = v51;
            }
          }
          v8 = v21;
          goto LABEL_50;
        }
        if ( (unsigned int)(unsigned __int8)v13 - 48 > 9 )
        {
          if ( v13 == 124 )
          {
            ++v54;
            ++v8;
            v6 = *(_QWORD *)(a1 + 64) + 56LL * v54 + 8;
          }
          else
          {
            v58 = &v60;
            if ( v13 == 94 )
            {
              v37 = *(_WORD *)(v8 + 1);
              v59 = 2;
              v60.m128i_i8[2] = 0;
              v60.m128i_i16[0] = v37;
              v38 = *(unsigned int *)(v6 + 8);
              if ( (unsigned int)v38 >= *(_DWORD *)(v6 + 12) )
              {
                sub_12BE710(v6, 0, v38, a4, a5, a6);
                LODWORD(v38) = *(_DWORD *)(v6 + 8);
              }
              v39 = (__m128i *)(*(_QWORD *)v6 + 32LL * (unsigned int)v38);
              if ( v39 )
              {
                v39->m128i_i64[0] = (__int64)v39[1].m128i_i64;
                if ( v58 == &v60 )
                {
                  v39[1] = _mm_load_si128(&v60);
                }
                else
                {
                  v39->m128i_i64[0] = (__int64)v58;
                  v39[1].m128i_i64[0] = v60.m128i_i64[0];
                }
                v39->m128i_i64[1] = v59;
                v59 = 0;
                v60.m128i_i8[0] = 0;
                ++*(_DWORD *)(v6 + 8);
              }
              else
              {
                v42 = v58;
                *(_DWORD *)(v6 + 8) = v38 + 1;
                if ( v42 != &v60 )
                  j_j___libc_free_0(v42, v60.m128i_i64[0] + 1);
              }
              v8 += 3;
            }
            else
            {
              v15 = *v8;
              v59 = 1;
              v60.m128i_i16[0] = v15;
              v16 = *(unsigned int *)(v6 + 8);
              if ( (unsigned int)v16 >= *(_DWORD *)(v6 + 12) )
              {
                sub_12BE710(v6, 0, v16, a4, a5, a6);
                LODWORD(v16) = *(_DWORD *)(v6 + 8);
              }
              v17 = (__m128i *)(*(_QWORD *)v6 + 32LL * (unsigned int)v16);
              if ( v17 )
              {
                v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
                if ( v58 == &v60 )
                {
                  v17[1] = _mm_load_si128(&v60);
                }
                else
                {
                  v17->m128i_i64[0] = (__int64)v58;
                  v17[1].m128i_i64[0] = v60.m128i_i64[0];
                }
                v17->m128i_i64[1] = v59;
                v59 = 0;
                v60.m128i_i8[0] = 0;
                ++*(_DWORD *)(v6 + 8);
              }
              else
              {
                v36 = v58;
                *(_DWORD *)(v6 + 8) = v16 + 1;
                if ( v36 != &v60 )
                  j_j___libc_free_0(v36, v60.m128i_i64[0] + 1);
              }
              ++v8;
            }
          }
          goto LABEL_50;
        }
        if ( v8 == (_BYTE *)v9 )
        {
          v58 = &v60;
          v57 = 0;
          goto LABEL_74;
        }
        a4 = (unsigned __int64)v8;
        do
          ++a4;
        while ( a4 != v9 && (unsigned int)*(unsigned __int8 *)a4 - 48 <= 9 );
        v58 = &v60;
        a6 = a4 - (_QWORD)v8;
        v57 = a4 - (_QWORD)v8;
        if ( a4 - (unsigned __int64)v8 > 0xF )
          break;
        if ( a6 != 1 )
        {
          if ( !a6 )
          {
            v8 = (_BYTE *)a4;
            goto LABEL_74;
          }
          v45 = &v60;
          goto LABEL_122;
        }
        v28 = *v8;
        v8 = (_BYTE *)a4;
        v60.m128i_i8[0] = v28;
LABEL_74:
        v59 = v57;
        v58->m128i_i8[v57] = 0;
        v29 = *(unsigned int *)(v6 + 8);
        if ( (unsigned int)v29 >= *(_DWORD *)(v6 + 12) )
        {
          sub_12BE710(v6, 0, v29, a4, a5, a6);
          LODWORD(v29) = *(_DWORD *)(v6 + 8);
        }
        v30 = *(_QWORD *)v6;
        v31 = (__m128i *)(*(_QWORD *)v6 + 32LL * (unsigned int)v29);
        if ( v31 )
        {
          v31->m128i_i64[0] = (__int64)v31[1].m128i_i64;
          if ( v58 == &v60 )
          {
            v31[1] = _mm_load_si128(&v60);
          }
          else
          {
            v31->m128i_i64[0] = (__int64)v58;
            v31[1].m128i_i64[0] = v60.m128i_i64[0];
          }
          v31->m128i_i64[1] = v59;
          v59 = 0;
          v60.m128i_i8[0] = 0;
          v30 = *(_QWORD *)v6;
          ++*(_DWORD *)(v6 + 8);
        }
        else
        {
          v40 = v58;
          *(_DWORD *)(v6 + 8) = v29 + 1;
          if ( v40 != &v60 )
          {
            j_j___libc_free_0(v40, v60.m128i_i64[0] + 1);
            v30 = *(_QWORD *)v6;
          }
        }
        v32 = strtol(*(const char **)(v30 + 32LL * *(unsigned int *)(v6 + 8) - 32), 0, 10);
        a4 = *(unsigned int *)(v46 + 8);
        if ( v32 >= a4 )
          goto LABEL_18;
        v33 = *(_QWORD *)v46 + 192LL * v32;
        if ( *(_DWORD *)v33 != 1 || *(_DWORD *)a1 )
          goto LABEL_18;
        if ( *(_BYTE *)(a1 + 11) )
        {
          if ( *(_DWORD *)(v33 + 72) <= v54 )
            goto LABEL_18;
          a4 = 7LL * v54;
          v34 = (_DWORD *)(*(_QWORD *)(v33 + 64) + 56LL * v54);
          if ( *v34 != -1 )
            goto LABEL_18;
          *v34 = *(_DWORD *)(v46 + 8);
        }
        else
        {
          v41 = *(int *)(v33 + 4);
          if ( (_DWORD)v41 != -1 && v41 != a4 )
            goto LABEL_18;
          *(_DWORD *)(v33 + 4) = a4;
        }
LABEL_50:
        if ( (_BYTE *)v9 == v8 )
          goto LABEL_120;
        v13 = *v8;
      }
      na = a4 - (_QWORD)v8;
      v52 = a4;
      v44 = sub_22409D0(&v58, &v57, 0);
      a4 = v52;
      a6 = na;
      v58 = (__m128i *)v44;
      v45 = (__m128i *)v44;
      v60.m128i_i64[0] = v57;
LABEL_122:
      v53 = (_BYTE *)a4;
      memcpy(v45, v8, a6);
      a4 = (unsigned __int64)v53;
      v8 = v53;
      goto LABEL_74;
    }
LABEL_120:
    LODWORD(a6) = 0;
  }
  return (unsigned int)a6;
}
