// Function: sub_16DA0A0
// Address: 0x16da0a0
//
__int64 __fastcall sub_16DA0A0(__int64 a1, __int64 a2, __int64 a3, __m128i *a4)
{
  __int64 v5; // r8
  __int64 v6; // r15
  __int64 v7; // rcx
  __int64 v9; // rbx
  __m128i *v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // r12
  __m128i *v13; // rax
  __int64 v14; // r12
  __m128i *v15; // rdi
  size_t v16; // rdx
  __m128i *v17; // rsi
  __int64 v18; // r13
  __m128i *p_src; // r8
  size_t v20; // rdx
  __m128i v21; // xmm0
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // r12
  __m128i *v25; // rdi
  __m128i *v26; // rax
  __m128i *v27; // r9
  size_t v28; // rdx
  __m128i *v29; // rax
  __m128i *v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // rsi
  __int64 result; // rax
  __m128i *v34; // rdi
  __int64 v35; // r12
  __m128i *v36; // rax
  __m128i *v37; // rcx
  __int64 v38; // rdx
  size_t v39; // rdx
  __int64 v40; // [rsp+0h] [rbp-80h]
  __m128i *v41; // [rsp+8h] [rbp-78h]
  __m128i *v42; // [rsp+8h] [rbp-78h]
  __int64 v43; // [rsp+10h] [rbp-70h]
  __int64 v44; // [rsp+10h] [rbp-70h]
  __m128i *v45; // [rsp+10h] [rbp-70h]
  __m128i *v46; // [rsp+18h] [rbp-68h]
  __m128i *v47; // [rsp+20h] [rbp-60h]
  size_t n; // [rsp+28h] [rbp-58h]
  __m128i src; // [rsp+30h] [rbp-50h] BYREF
  __m128i v50; // [rsp+40h] [rbp-40h]

  v5 = a3;
  v6 = a2;
  v7 = (a3 - 1) / 2;
  v9 = a1 + 48 * a2;
  v10 = (__m128i *)(v9 + 16);
  if ( a2 < v7 )
  {
    while ( 1 )
    {
      v18 = 2 * (a2 + 1);
      v9 = a1 + 96 * (a2 + 1);
      if ( *(_QWORD *)(a1 + 48 * (v18 - 1) + 40) < *(_QWORD *)(v9 + 40) )
        v9 = a1 + 48 * --v18;
      v12 = 3 * a2;
      v13 = *(__m128i **)v9;
      v17 = (__m128i *)(v9 + 16);
      v14 = a1 + 16 * v12;
      v15 = *(__m128i **)v14;
      if ( *(_QWORD *)v9 == v9 + 16 )
      {
        v16 = *(_QWORD *)(v9 + 8);
        if ( v16 )
        {
          if ( v16 == 1 )
          {
            v15->m128i_i8[0] = *(_BYTE *)(v9 + 16);
            v16 = *(_QWORD *)(v9 + 8);
            v15 = *(__m128i **)v14;
          }
          else
          {
            v40 = v7;
            v41 = a4;
            v43 = v5;
            memcpy(v15, v17, v16);
            v16 = *(_QWORD *)(v9 + 8);
            v15 = *(__m128i **)v14;
            v7 = v40;
            a4 = v41;
            v5 = v43;
            v17 = (__m128i *)(v9 + 16);
          }
        }
        *(_QWORD *)(v14 + 8) = v16;
        v15->m128i_i8[v16] = 0;
        v15 = *(__m128i **)v9;
      }
      else
      {
        if ( v15 == v10 )
        {
          *(_QWORD *)v14 = v13;
          *(_QWORD *)(v14 + 8) = *(_QWORD *)(v9 + 8);
          *(_QWORD *)(v14 + 16) = *(_QWORD *)(v9 + 16);
        }
        else
        {
          *(_QWORD *)v14 = v13;
          v11 = *(_QWORD *)(v14 + 16);
          *(_QWORD *)(v14 + 8) = *(_QWORD *)(v9 + 8);
          *(_QWORD *)(v14 + 16) = *(_QWORD *)(v9 + 16);
          if ( v15 )
          {
            *(_QWORD *)v9 = v15;
            *(_QWORD *)(v9 + 16) = v11;
            goto LABEL_6;
          }
        }
        *(_QWORD *)v9 = v17;
        v15 = (__m128i *)(v9 + 16);
      }
LABEL_6:
      *(_QWORD *)(v9 + 8) = 0;
      v15->m128i_i8[0] = 0;
      *(_QWORD *)(v14 + 32) = *(_QWORD *)(v9 + 32);
      *(_QWORD *)(v14 + 40) = *(_QWORD *)(v9 + 40);
      if ( v18 >= v7 )
        goto LABEL_17;
      v10 = v17;
      a2 = v18;
    }
  }
  v17 = (__m128i *)(v9 + 16);
  v18 = v6;
LABEL_17:
  if ( (v5 & 1) != 0 || (v5 - 2) / 2 != v18 )
    goto LABEL_19;
  v18 = 2 * v18 + 1;
  v34 = *(__m128i **)v9;
  v35 = a1 + 48 * v18;
  v36 = *(__m128i **)v35;
  v37 = (__m128i *)(v35 + 16);
  if ( *(_QWORD *)v35 == v35 + 16 )
  {
    v39 = *(_QWORD *)(v35 + 8);
    if ( v39 )
    {
      if ( v39 == 1 )
      {
        v34->m128i_i8[0] = *(_BYTE *)(v35 + 16);
        v39 = *(_QWORD *)(v35 + 8);
        v34 = *(__m128i **)v9;
      }
      else
      {
        v45 = a4;
        memcpy(v34, (const void *)(v35 + 16), v39);
        v39 = *(_QWORD *)(v35 + 8);
        v34 = *(__m128i **)v9;
        a4 = v45;
        v37 = (__m128i *)(v35 + 16);
      }
    }
    *(_QWORD *)(v9 + 8) = v39;
    v34->m128i_i8[v39] = 0;
    v34 = *(__m128i **)v35;
    goto LABEL_58;
  }
  if ( v34 == v17 )
  {
    *(_QWORD *)v9 = v36;
    *(_QWORD *)(v9 + 8) = *(_QWORD *)(v35 + 8);
    *(_QWORD *)(v9 + 16) = *(_QWORD *)(v35 + 16);
    goto LABEL_61;
  }
  *(_QWORD *)v9 = v36;
  v38 = *(_QWORD *)(v9 + 16);
  *(_QWORD *)(v9 + 8) = *(_QWORD *)(v35 + 8);
  *(_QWORD *)(v9 + 16) = *(_QWORD *)(v35 + 16);
  if ( !v34 )
  {
LABEL_61:
    *(_QWORD *)v35 = v37;
    v34 = (__m128i *)(v35 + 16);
    goto LABEL_58;
  }
  *(_QWORD *)v35 = v34;
  *(_QWORD *)(v35 + 16) = v38;
LABEL_58:
  *(_QWORD *)(v35 + 8) = 0;
  v17 = v37;
  v34->m128i_i8[0] = 0;
  *(_QWORD *)(v9 + 32) = *(_QWORD *)(v35 + 32);
  *(_QWORD *)(v9 + 40) = *(_QWORD *)(v35 + 40);
  v9 = a1 + 48 * v18;
LABEL_19:
  p_src = &src;
  v47 = &src;
  if ( (__m128i *)a4->m128i_i64[0] == &a4[1] )
  {
    src = _mm_loadu_si128(a4 + 1);
  }
  else
  {
    v47 = (__m128i *)a4->m128i_i64[0];
    src.m128i_i64[0] = a4[1].m128i_i64[0];
  }
  v20 = a4->m128i_u64[1];
  v21 = _mm_loadu_si128(a4 + 2);
  a4->m128i_i64[0] = (__int64)a4[1].m128i_i64;
  a4->m128i_i64[1] = 0;
  n = v20;
  a4[1].m128i_i8[0] = 0;
  v50 = v21;
  v22 = (v18 - 1) / 2;
  if ( v18 <= v6 )
  {
LABEL_37:
    v29 = v47;
    v30 = *(__m128i **)v9;
    if ( v47 == p_src )
      goto LABEL_47;
    goto LABEL_38;
  }
  while ( 1 )
  {
    v24 = a1 + 48 * v22;
    v9 = a1 + 48 * v18;
    v25 = *(__m128i **)v9;
    if ( v50.m128i_i64[1] >= *(_QWORD *)(v24 + 40) )
    {
      v20 = n;
      goto LABEL_37;
    }
    v26 = *(__m128i **)v24;
    v27 = (__m128i *)(v24 + 16);
    if ( *(_QWORD *)v24 == v24 + 16 )
    {
      v28 = *(_QWORD *)(v24 + 8);
      if ( v28 )
      {
        if ( v28 == 1 )
        {
          v25->m128i_i8[0] = *(_BYTE *)(v24 + 16);
          v28 = *(_QWORD *)(v24 + 8);
          v25 = *(__m128i **)v9;
        }
        else
        {
          v42 = p_src;
          v44 = v22;
          memcpy(v25, (const void *)(v24 + 16), v28);
          v28 = *(_QWORD *)(v24 + 8);
          v25 = *(__m128i **)v9;
          p_src = v42;
          v22 = v44;
          v27 = (__m128i *)(v24 + 16);
        }
      }
      *(_QWORD *)(v9 + 8) = v28;
      v25->m128i_i8[v28] = 0;
      v25 = *(__m128i **)v24;
    }
    else
    {
      if ( v17 == v25 )
      {
        *(_QWORD *)v9 = v26;
        *(_QWORD *)(v9 + 8) = *(_QWORD *)(v24 + 8);
        *(_QWORD *)(v9 + 16) = *(_QWORD *)(v24 + 16);
      }
      else
      {
        *(_QWORD *)v9 = v26;
        v23 = *(_QWORD *)(v9 + 16);
        *(_QWORD *)(v9 + 8) = *(_QWORD *)(v24 + 8);
        *(_QWORD *)(v9 + 16) = *(_QWORD *)(v24 + 16);
        if ( v25 )
        {
          *(_QWORD *)v24 = v25;
          *(_QWORD *)(v24 + 16) = v23;
          goto LABEL_26;
        }
      }
      *(_QWORD *)v24 = v27;
      v25 = (__m128i *)(v24 + 16);
    }
LABEL_26:
    *(_QWORD *)(v24 + 8) = 0;
    v18 = v22;
    v25->m128i_i8[0] = 0;
    *(_QWORD *)(v9 + 32) = *(_QWORD *)(v24 + 32);
    *(_QWORD *)(v9 + 40) = *(_QWORD *)(v24 + 40);
    if ( v6 >= v22 )
      break;
    v17 = v27;
    v22 = (v22 - 1) / 2;
  }
  v29 = v47;
  v9 = v24;
  v20 = n;
  v17 = v27;
  v30 = *(__m128i **)v24;
  if ( v47 == p_src )
  {
LABEL_47:
    if ( v20 )
    {
      if ( v20 == 1 )
      {
        v30->m128i_i8[0] = src.m128i_i8[0];
        v20 = n;
        v30 = *(__m128i **)v9;
      }
      else
      {
        v46 = p_src;
        memcpy(v30, p_src, v20);
        v20 = n;
        v30 = *(__m128i **)v9;
        p_src = v46;
      }
    }
    *(_QWORD *)(v9 + 8) = v20;
    v30->m128i_i8[v20] = 0;
    v30 = v47;
    goto LABEL_41;
  }
LABEL_38:
  v31 = src.m128i_i64[0];
  if ( v30 == v17 )
  {
    *(_QWORD *)v9 = v29;
    *(_QWORD *)(v9 + 8) = v20;
    *(_QWORD *)(v9 + 16) = v31;
  }
  else
  {
    v32 = *(_QWORD *)(v9 + 16);
    *(_QWORD *)v9 = v29;
    *(_QWORD *)(v9 + 8) = v20;
    *(_QWORD *)(v9 + 16) = v31;
    if ( v30 )
    {
      v47 = v30;
      src.m128i_i64[0] = v32;
      goto LABEL_41;
    }
  }
  v47 = p_src;
  p_src = &src;
  v30 = &src;
LABEL_41:
  v30->m128i_i8[0] = 0;
  *(_QWORD *)(v9 + 32) = v50.m128i_i64[0];
  result = v50.m128i_i64[1];
  *(_QWORD *)(v9 + 40) = v50.m128i_i64[1];
  if ( v47 != p_src )
    return j_j___libc_free_0(v47, src.m128i_i64[0] + 1);
  return result;
}
