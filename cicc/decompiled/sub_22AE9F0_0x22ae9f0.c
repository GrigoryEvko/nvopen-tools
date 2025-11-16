// Function: sub_22AE9F0
// Address: 0x22ae9f0
//
void __fastcall sub_22AE9F0(__int64 a1, char a2)
{
  __int64 v3; // r12
  unsigned __int64 *v4; // rdi
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r15
  unsigned int v8; // r12d
  __int64 *v9; // rax
  __m128i *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned __int64 v13; // rdx
  const char *v14; // r14
  size_t v15; // r13
  __int64 v16; // rax
  __m128i *v17; // rdx
  __m128i v18; // rax
  bool v19; // zf
  unsigned __int64 v20; // rdx
  char *v21; // r14
  size_t v22; // r13
  __int64 v23; // rax
  __m128i *v24; // rdx
  __m128i *v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rax
  unsigned __int64 v29; // rdi
  __m128i *v30; // rdi
  __m128i *v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rdi
  __int64 v34; // rcx
  __m128i *v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // rcx
  __int64 v39; // rdi
  __m128i *v40; // rdi
  __int64 v41; // [rsp+8h] [rbp-58h] BYREF
  __m128i v42; // [rsp+10h] [rbp-50h] BYREF
  __m128i src[4]; // [rsp+20h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a1 + 16);
  if ( *(_BYTE *)v3 != 85 )
    v3 = 0;
  v4 = (unsigned __int64 *)(a1 + 88);
  if ( *(_BYTE *)(a1 + 120) )
  {
    sub_2241130(v4, 0, *(_QWORD *)(a1 + 96), byte_3F871B3, 0);
    v5 = *(_QWORD *)(a1 + 16);
    if ( *(_BYTE *)v5 != 85 )
      goto LABEL_5;
  }
  else
  {
    *(_QWORD *)(a1 + 88) = a1 + 104;
    sub_22AD7C0((__int64 *)v4, byte_3F871B3, (__int64)byte_3F871B3);
    v5 = *(_QWORD *)(a1 + 16);
    *(_BYTE *)(a1 + 120) = 1;
    if ( *(_BYTE *)v5 != 85 )
    {
LABEL_5:
      if ( sub_B491E0(v3) || !a2 )
        return;
      v12 = *(_QWORD *)(v3 - 32);
      if ( v12 )
      {
        if ( *(_BYTE *)v12 )
        {
          v12 = 0;
        }
        else if ( *(_QWORD *)(v12 + 24) != *(_QWORD *)(v3 + 80) )
        {
          v12 = 0;
        }
      }
      v14 = sub_BD5D20(v12);
      v15 = v13;
      if ( v14 )
      {
        v41 = v13;
        v16 = v13;
        v42.m128i_i64[0] = (__int64)src;
        if ( v13 > 0xF )
        {
          v42.m128i_i64[0] = sub_22409D0((__int64)&v42, (unsigned __int64 *)&v41, 0);
          v35 = (__m128i *)v42.m128i_i64[0];
          src[0].m128i_i64[0] = v41;
        }
        else
        {
          if ( v13 == 1 )
          {
            src[0].m128i_i8[0] = *v14;
            v17 = src;
            goto LABEL_26;
          }
          if ( !v13 )
          {
            v17 = src;
            goto LABEL_26;
          }
          v35 = src;
        }
        memcpy(v35, v14, v15);
        v16 = v41;
        v17 = (__m128i *)v42.m128i_i64[0];
LABEL_26:
        v42.m128i_i64[1] = v16;
        v17->m128i_i8[v16] = 0;
        if ( !*(_BYTE *)(a1 + 120) )
        {
          v18 = v42;
          *(_QWORD *)(a1 + 88) = a1 + 104;
          if ( (__m128i *)v18.m128i_i64[0] != src )
          {
LABEL_28:
            *(_QWORD *)(a1 + 88) = v18.m128i_i64[0];
            *(_QWORD *)(a1 + 104) = src[0].m128i_i64[0];
LABEL_32:
            *(_QWORD *)(a1 + 96) = v18.m128i_i64[1];
            *(_BYTE *)(a1 + 120) = 1;
            return;
          }
LABEL_31:
          *(__m128i *)(a1 + 104) = _mm_load_si128(src);
          goto LABEL_32;
        }
        v30 = *(__m128i **)(a1 + 88);
        v31 = v30;
        if ( (__m128i *)v42.m128i_i64[0] != src )
        {
          v32 = src[0].m128i_i64[0];
          v33 = v42.m128i_i64[1];
          if ( v31 == (__m128i *)(a1 + 104) )
          {
            *(_QWORD *)(a1 + 88) = v42.m128i_i64[0];
            *(_QWORD *)(a1 + 96) = v33;
            *(_QWORD *)(a1 + 104) = v32;
          }
          else
          {
            v34 = *(_QWORD *)(a1 + 104);
            *(_QWORD *)(a1 + 88) = v42.m128i_i64[0];
            *(_QWORD *)(a1 + 96) = v33;
            *(_QWORD *)(a1 + 104) = v32;
            if ( v31 )
            {
              v42.m128i_i64[0] = (__int64)v31;
              src[0].m128i_i64[0] = v34;
              goto LABEL_55;
            }
          }
          goto LABEL_71;
        }
LABEL_72:
        v36 = v42.m128i_i64[1];
        if ( v42.m128i_i64[1] )
        {
          if ( v42.m128i_i64[1] == 1 )
            v30->m128i_i8[0] = src[0].m128i_i8[0];
          else
            memcpy(v30, src, v42.m128i_u64[1]);
          v36 = v42.m128i_i64[1];
          v30 = *(__m128i **)(a1 + 88);
        }
        goto LABEL_60;
      }
      v19 = *(_BYTE *)(a1 + 120) == 0;
      src[0].m128i_i8[0] = 0;
      v42.m128i_i64[0] = (__int64)src;
      if ( v19 )
      {
        v18.m128i_i64[1] = 0;
        *(_QWORD *)(a1 + 88) = a1 + 104;
        goto LABEL_31;
      }
LABEL_59:
      v30 = *(__m128i **)(a1 + 88);
      v36 = 0;
LABEL_60:
      *(_QWORD *)(a1 + 96) = v36;
      v30->m128i_i8[v36] = 0;
      v31 = (__m128i *)v42.m128i_i64[0];
      goto LABEL_55;
    }
  }
  v6 = *(_QWORD *)(v5 - 32);
  if ( !v6 )
    goto LABEL_5;
  if ( *(_BYTE *)v6 )
    goto LABEL_5;
  v7 = *(_QWORD *)(v5 + 80);
  if ( *(_QWORD *)(v6 + 24) != v7 || (*(_BYTE *)(v6 + 33) & 0x20) == 0 )
    goto LABEL_5;
  v8 = *(_DWORD *)(v6 + 36);
  if ( !(unsigned __int8)sub_B60C20(v8) )
  {
    v21 = sub_B60C10(v8);
    v22 = v20;
    if ( !v21 )
    {
      v19 = *(_BYTE *)(a1 + 120) == 0;
      src[0].m128i_i8[0] = 0;
      v42.m128i_i64[0] = (__int64)src;
      if ( v19 )
      {
        v18.m128i_i64[1] = 0;
        *(_QWORD *)(a1 + 88) = a1 + 104;
        goto LABEL_31;
      }
      goto LABEL_59;
    }
    v41 = v20;
    v23 = v20;
    v42.m128i_i64[0] = (__int64)src;
    if ( v20 > 0xF )
    {
      v42.m128i_i64[0] = sub_22409D0((__int64)&v42, (unsigned __int64 *)&v41, 0);
      v40 = (__m128i *)v42.m128i_i64[0];
      src[0].m128i_i64[0] = v41;
    }
    else
    {
      if ( v20 == 1 )
      {
        src[0].m128i_i8[0] = *v21;
        v24 = src;
        goto LABEL_39;
      }
      if ( !v20 )
      {
        v24 = src;
        goto LABEL_39;
      }
      v40 = src;
    }
    memcpy(v40, v21, v22);
    v23 = v41;
    v24 = (__m128i *)v42.m128i_i64[0];
LABEL_39:
    v42.m128i_i64[1] = v23;
    v24->m128i_i8[v23] = 0;
    if ( !*(_BYTE *)(a1 + 120) )
    {
      v18 = v42;
      *(_QWORD *)(a1 + 88) = a1 + 104;
      if ( (__m128i *)v18.m128i_i64[0] != src )
        goto LABEL_28;
      goto LABEL_31;
    }
    v30 = *(__m128i **)(a1 + 88);
    v31 = v30;
    if ( (__m128i *)v42.m128i_i64[0] != src )
    {
      v37 = v42.m128i_i64[1];
      v38 = src[0].m128i_i64[0];
      if ( v30 == (__m128i *)(a1 + 104) )
      {
        *(_QWORD *)(a1 + 88) = v42.m128i_i64[0];
        *(_QWORD *)(a1 + 96) = v37;
        *(_QWORD *)(a1 + 104) = v38;
      }
      else
      {
        v39 = *(_QWORD *)(a1 + 104);
        *(_QWORD *)(a1 + 88) = v42.m128i_i64[0];
        *(_QWORD *)(a1 + 96) = v37;
        *(_QWORD *)(a1 + 104) = v38;
        if ( v31 )
        {
          v42.m128i_i64[0] = (__int64)v31;
          src[0].m128i_i64[0] = v39;
          goto LABEL_55;
        }
      }
LABEL_71:
      v42.m128i_i64[0] = (__int64)src;
      v31 = src;
LABEL_55:
      v42.m128i_i64[1] = 0;
      v31->m128i_i8[0] = 0;
      v29 = v42.m128i_i64[0];
      if ( (__m128i *)v42.m128i_i64[0] == src )
        return;
LABEL_50:
      j_j___libc_free_0(v29);
      return;
    }
    goto LABEL_72;
  }
  v9 = (__int64 *)sub_B43CA0(v5);
  sub_B6E0E0(&v42, v8, *(_QWORD *)(v7 + 16) + 8LL, (8LL * *(unsigned int *)(v7 + 12) - 8) >> 3, v9, v7);
  v10 = (__m128i *)(a1 + 104);
  if ( *(_BYTE *)(a1 + 120) )
  {
    v25 = *(__m128i **)(a1 + 88);
    v26 = v42.m128i_i64[1];
    if ( (__m128i *)v42.m128i_i64[0] == src )
    {
      if ( v42.m128i_i64[1] )
      {
        if ( v42.m128i_i64[1] == 1 )
          v25->m128i_i8[0] = src[0].m128i_i8[0];
        else
          memcpy(v25, src, v42.m128i_u64[1]);
        v26 = v42.m128i_i64[1];
        v25 = *(__m128i **)(a1 + 88);
      }
      *(_QWORD *)(a1 + 96) = v26;
      v25->m128i_i8[v26] = 0;
      v25 = (__m128i *)v42.m128i_i64[0];
    }
    else
    {
      v27 = src[0].m128i_i64[0];
      if ( v25 == v10 )
      {
        *(_QWORD *)(a1 + 88) = v42.m128i_i64[0];
        *(_QWORD *)(a1 + 96) = v26;
        *(_QWORD *)(a1 + 104) = v27;
      }
      else
      {
        v28 = *(_QWORD *)(a1 + 104);
        *(_QWORD *)(a1 + 88) = v42.m128i_i64[0];
        *(_QWORD *)(a1 + 96) = v26;
        *(_QWORD *)(a1 + 104) = v27;
        if ( v25 )
        {
          v42.m128i_i64[0] = (__int64)v25;
          src[0].m128i_i64[0] = v28;
          goto LABEL_49;
        }
      }
      v42.m128i_i64[0] = (__int64)src;
      v25 = src;
    }
LABEL_49:
    v42.m128i_i64[1] = 0;
    v25->m128i_i8[0] = 0;
    v29 = v42.m128i_i64[0];
    if ( (__m128i *)v42.m128i_i64[0] == src )
      return;
    goto LABEL_50;
  }
  *(_QWORD *)(a1 + 88) = v10;
  if ( (__m128i *)v42.m128i_i64[0] == src )
  {
    *(__m128i *)(a1 + 104) = _mm_load_si128(src);
  }
  else
  {
    *(_QWORD *)(a1 + 88) = v42.m128i_i64[0];
    *(_QWORD *)(a1 + 104) = src[0].m128i_i64[0];
  }
  v11 = v42.m128i_i64[1];
  *(_BYTE *)(a1 + 120) = 1;
  *(_QWORD *)(a1 + 96) = v11;
}
