// Function: sub_1E937F0
// Address: 0x1e937f0
//
__int64 __fastcall sub_1E937F0(__int64 a1, __int64 a2, __int64 a3, __m128i *a4)
{
  __int64 v4; // r11
  __int64 v5; // rbx
  __int64 v6; // r14
  _BYTE *v7; // r15
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rcx
  __int64 v11; // r9
  unsigned __int64 v12; // r13
  const void *v13; // r8
  size_t v14; // r10
  const void *v15; // rsi
  size_t v16; // rdx
  int v17; // eax
  __int64 v18; // r13
  __int64 v19; // r13
  _BYTE *v20; // r10
  size_t v21; // rdx
  __m128i *v22; // r9
  __m128i *v23; // r14
  __int64 v24; // rax
  size_t v25; // r15
  __int64 v26; // r13
  size_t v27; // rax
  __m128i *v28; // rsi
  __int64 v29; // r15
  size_t v30; // r12
  __int64 v31; // rax
  size_t v32; // rdx
  size_t v33; // r14
  const void *v34; // rcx
  signed __int64 v35; // rax
  __int64 v36; // r12
  __m128i *v37; // rdi
  size_t v38; // rax
  __int64 v39; // r12
  __m128i *v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 result; // rax
  __m128i *v44; // rdi
  __int64 v45; // r13
  __m128i *v46; // rax
  __int64 v47; // rdx
  size_t v48; // rdx
  __int64 v51; // [rsp+18h] [rbp-A8h]
  __int64 v53; // [rsp+28h] [rbp-98h]
  void *src; // [rsp+30h] [rbp-90h]
  __int64 v55; // [rsp+38h] [rbp-88h]
  __int64 v56; // [rsp+38h] [rbp-88h]
  __m128i *v57; // [rsp+40h] [rbp-80h]
  const void *v58; // [rsp+50h] [rbp-70h]
  __int64 v59; // [rsp+50h] [rbp-70h]
  const void *v60; // [rsp+50h] [rbp-70h]
  __int64 v61; // [rsp+50h] [rbp-70h]
  __int64 v62; // [rsp+50h] [rbp-70h]
  __m128i *s2; // [rsp+60h] [rbp-60h]
  size_t v64; // [rsp+68h] [rbp-58h]
  __m128i v65; // [rsp+70h] [rbp-50h] BYREF
  __int64 v66; // [rsp+80h] [rbp-40h]

  v4 = a1;
  v51 = (a3 - 1) / 2;
  v5 = a1 + 40 * a2;
  if ( v51 <= a2 )
  {
    v9 = a2;
    v22 = (__m128i *)(v5 + 16);
    goto LABEL_25;
  }
  v6 = a2;
  v7 = (_BYTE *)(v5 + 16);
  while ( 1 )
  {
    v9 = 2 * (v6 + 1);
    v10 = v9 - 1;
    v5 = v4 + 80 * (v6 + 1);
    v11 = v4 + 40 * (v9 - 1);
    v12 = *(_QWORD *)(v5 + 8);
    v13 = *(const void **)v5;
    v14 = *(_QWORD *)(v11 + 8);
    v15 = *(const void **)v11;
    v16 = v14;
    if ( v12 <= v14 )
      v16 = *(_QWORD *)(v5 + 8);
    if ( !v16 )
      goto LABEL_12;
    v53 = v4;
    src = *(void **)(v11 + 8);
    v55 = v4 + 40 * (v9 - 1);
    v58 = *(const void **)v5;
    v17 = memcmp(*(const void **)v5, v15, v16);
    v13 = v58;
    v10 = v9 - 1;
    v11 = v55;
    v14 = (size_t)src;
    v4 = v53;
    if ( !v17 )
    {
LABEL_12:
      v18 = v12 - v14;
      if ( v18 >= 0x80000000LL )
        goto LABEL_17;
      if ( v18 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v13 = v15;
        v5 = v11;
        v9 = v10;
        goto LABEL_17;
      }
      v17 = v18;
    }
    if ( v17 < 0 )
    {
      v13 = v15;
      v5 = v11;
      v9 = v10;
    }
LABEL_17:
    v19 = v4 + 40 * v6;
    v20 = *(_BYTE **)v19;
    if ( (const void *)(v5 + 16) == v13 )
    {
      v21 = *(_QWORD *)(v5 + 8);
      if ( v21 )
      {
        if ( v21 == 1 )
        {
          *v20 = *(_BYTE *)(v5 + 16);
          v21 = *(_QWORD *)(v5 + 8);
          v20 = *(_BYTE **)v19;
        }
        else
        {
          v59 = v4;
          memcpy(*(void **)v19, v13, v21);
          v21 = *(_QWORD *)(v5 + 8);
          v20 = *(_BYTE **)v19;
          v4 = v59;
        }
      }
      *(_QWORD *)(v19 + 8) = v21;
      v20[v21] = 0;
      v20 = *(_BYTE **)v5;
    }
    else
    {
      if ( v20 == v7 )
      {
        *(_QWORD *)v19 = v13;
        *(_QWORD *)(v19 + 8) = *(_QWORD *)(v5 + 8);
        *(_QWORD *)(v19 + 16) = *(_QWORD *)(v5 + 16);
      }
      else
      {
        *(_QWORD *)v19 = v13;
        v8 = *(_QWORD *)(v19 + 16);
        *(_QWORD *)(v19 + 8) = *(_QWORD *)(v5 + 8);
        *(_QWORD *)(v19 + 16) = *(_QWORD *)(v5 + 16);
        if ( v20 )
        {
          *(_QWORD *)v5 = v20;
          *(_QWORD *)(v5 + 16) = v8;
          goto LABEL_6;
        }
      }
      *(_QWORD *)v5 = v5 + 16;
      v20 = (_BYTE *)(v5 + 16);
    }
LABEL_6:
    *(_QWORD *)(v5 + 8) = 0;
    *v20 = 0;
    *(_QWORD *)(v19 + 32) = *(_QWORD *)(v5 + 32);
    if ( v9 >= v51 )
      break;
    v7 = (_BYTE *)(v5 + 16);
    v6 = v9;
  }
  v22 = (__m128i *)(v5 + 16);
LABEL_25:
  if ( (a3 & 1) == 0 && v9 == (a3 - 2) / 2 )
  {
    v9 = 2 * v9 + 1;
    v44 = *(__m128i **)v5;
    v45 = v4 + 40 * v9;
    v46 = *(__m128i **)v45;
    if ( *(_QWORD *)v45 == v45 + 16 )
    {
      v48 = *(_QWORD *)(v45 + 8);
      if ( v48 )
      {
        if ( v48 == 1 )
        {
          v44->m128i_i8[0] = *(_BYTE *)(v45 + 16);
          v48 = *(_QWORD *)(v45 + 8);
          v44 = *(__m128i **)v5;
        }
        else
        {
          v62 = v4;
          memcpy(v44, (const void *)(v45 + 16), v48);
          v48 = *(_QWORD *)(v45 + 8);
          v44 = *(__m128i **)v5;
          v4 = v62;
        }
      }
      *(_QWORD *)(v5 + 8) = v48;
      v44->m128i_i8[v48] = 0;
      v44 = *(__m128i **)v45;
      goto LABEL_73;
    }
    if ( v44 == v22 )
    {
      *(_QWORD *)v5 = v46;
      *(_QWORD *)(v5 + 8) = *(_QWORD *)(v45 + 8);
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(v45 + 16);
    }
    else
    {
      *(_QWORD *)v5 = v46;
      v47 = *(_QWORD *)(v5 + 16);
      *(_QWORD *)(v5 + 8) = *(_QWORD *)(v45 + 8);
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(v45 + 16);
      if ( v44 )
      {
        *(_QWORD *)v45 = v44;
        *(_QWORD *)(v45 + 16) = v47;
LABEL_73:
        *(_QWORD *)(v45 + 8) = 0;
        v22 = (__m128i *)(v45 + 16);
        v44->m128i_i8[0] = 0;
        *(_QWORD *)(v5 + 32) = *(_QWORD *)(v45 + 32);
        v5 = v45;
        goto LABEL_27;
      }
    }
    *(_QWORD *)v45 = v45 + 16;
    v44 = (__m128i *)(v45 + 16);
    goto LABEL_73;
  }
LABEL_27:
  s2 = &v65;
  v23 = (__m128i *)a4->m128i_i64[0];
  if ( (__m128i *)a4->m128i_i64[0] == &a4[1] )
  {
    v23 = &v65;
    v65 = _mm_loadu_si128(a4 + 1);
  }
  else
  {
    s2 = (__m128i *)a4->m128i_i64[0];
    v65.m128i_i64[0] = a4[1].m128i_i64[0];
  }
  a4->m128i_i64[0] = (__int64)a4[1].m128i_i64;
  v24 = a4[2].m128i_i64[0];
  v25 = a4->m128i_u64[1];
  a4[1].m128i_i8[0] = 0;
  v66 = v24;
  v64 = v25;
  a4->m128i_i64[1] = 0;
  v26 = (v9 - 1) / 2;
  if ( v9 <= a2 )
  {
LABEL_51:
    v40 = *(__m128i **)v5;
    if ( v23 == &v65 )
      goto LABEL_61;
    goto LABEL_52;
  }
  v27 = v25;
  v28 = v23;
  v29 = v9;
  v30 = v27;
  while ( 2 )
  {
    v32 = v30;
    v5 = v4 + 40 * v26;
    v33 = *(_QWORD *)(v5 + 8);
    v34 = *(const void **)v5;
    if ( v33 <= v30 )
      v32 = *(_QWORD *)(v5 + 8);
    if ( v32 )
    {
      v56 = v4;
      v57 = v22;
      v60 = *(const void **)v5;
      LODWORD(v35) = memcmp(*(const void **)v5, v28, v32);
      v34 = v60;
      v22 = v57;
      v4 = v56;
      if ( (_DWORD)v35 )
        goto LABEL_42;
    }
    v35 = v33 - v30;
    if ( (__int64)(v33 - v30) >= 0x80000000LL )
      goto LABEL_50;
    if ( v35 > (__int64)0xFFFFFFFF7FFFFFFFLL )
    {
LABEL_42:
      if ( (int)v35 >= 0 )
      {
LABEL_50:
        v38 = v30;
        v39 = v29;
        v23 = v28;
        v25 = v38;
        v5 = v4 + 40 * v39;
        goto LABEL_51;
      }
    }
    v36 = v4 + 40 * v29;
    v37 = *(__m128i **)v36;
    if ( v34 == (const void *)(v5 + 16) )
    {
      if ( v33 )
      {
        if ( v33 == 1 )
        {
          v37->m128i_i8[0] = *(_BYTE *)(v5 + 16);
          v33 = *(_QWORD *)(v5 + 8);
          v37 = *(__m128i **)v36;
        }
        else
        {
          v61 = v4;
          memcpy(v37, (const void *)(v5 + 16), v33);
          v33 = *(_QWORD *)(v5 + 8);
          v37 = *(__m128i **)v36;
          v4 = v61;
        }
      }
      *(_QWORD *)(v36 + 8) = v33;
      v37->m128i_i8[v33] = 0;
      v37 = *(__m128i **)v5;
    }
    else
    {
      if ( v37 == v22 )
      {
        *(_QWORD *)v36 = v34;
        *(_QWORD *)(v36 + 8) = *(_QWORD *)(v5 + 8);
        *(_QWORD *)(v36 + 16) = *(_QWORD *)(v5 + 16);
      }
      else
      {
        *(_QWORD *)v36 = v34;
        v31 = *(_QWORD *)(v36 + 16);
        *(_QWORD *)(v36 + 8) = *(_QWORD *)(v5 + 8);
        *(_QWORD *)(v36 + 16) = *(_QWORD *)(v5 + 16);
        if ( v37 )
        {
          *(_QWORD *)v5 = v37;
          *(_QWORD *)(v5 + 16) = v31;
          goto LABEL_34;
        }
      }
      *(_QWORD *)v5 = v5 + 16;
      v37 = (__m128i *)(v5 + 16);
    }
LABEL_34:
    *(_QWORD *)(v5 + 8) = 0;
    v37->m128i_i8[0] = 0;
    *(_QWORD *)(v36 + 32) = *(_QWORD *)(v5 + 32);
    if ( a2 < v26 )
    {
      v28 = s2;
      v30 = v64;
      v22 = (__m128i *)(v5 + 16);
      v29 = v26;
      v26 = (v26 - 1) / 2;
      continue;
    }
    break;
  }
  v23 = s2;
  v25 = v64;
  v40 = *(__m128i **)v5;
  v22 = (__m128i *)(v5 + 16);
  if ( s2 == &v65 )
  {
LABEL_61:
    if ( v25 )
    {
      if ( v25 == 1 )
        v40->m128i_i8[0] = v65.m128i_i8[0];
      else
        memcpy(v40, &v65, v25);
      v25 = v64;
      v40 = *(__m128i **)v5;
    }
    *(_QWORD *)(v5 + 8) = v25;
    v40->m128i_i8[v25] = 0;
    v40 = s2;
    goto LABEL_55;
  }
LABEL_52:
  v41 = v65.m128i_i64[0];
  if ( v22 == v40 )
  {
    *(_QWORD *)v5 = v23;
    *(_QWORD *)(v5 + 8) = v25;
    *(_QWORD *)(v5 + 16) = v41;
  }
  else
  {
    v42 = *(_QWORD *)(v5 + 16);
    *(_QWORD *)v5 = v23;
    *(_QWORD *)(v5 + 8) = v25;
    *(_QWORD *)(v5 + 16) = v41;
    if ( v40 )
    {
      s2 = v40;
      v65.m128i_i64[0] = v42;
      goto LABEL_55;
    }
  }
  s2 = &v65;
  v40 = &v65;
LABEL_55:
  v40->m128i_i8[0] = 0;
  result = v66;
  *(_QWORD *)(v5 + 32) = v66;
  if ( s2 != &v65 )
    return j_j___libc_free_0(s2, v65.m128i_i64[0] + 1);
  return result;
}
