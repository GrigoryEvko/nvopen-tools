// Function: sub_CA95D0
// Address: 0xca95d0
//
__m128i *__fastcall sub_CA95D0(__m128i *a1, __int64 a2)
{
  unsigned __int64 v3; // rbx
  _WORD *v4; // r13
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r15
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rdx
  _BYTE *v13; // rsi
  __int64 v14; // rcx
  __int8 *v15; // rdi
  __int64 v16; // rsi
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdx
  void *v19; // rax
  unsigned int v20; // eax
  __m128i *v21; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  _BYTE *v25; // rsi
  __int64 v26; // rcx
  __int8 *v27; // rdi
  __int64 v28; // rsi
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rdx
  _BYTE *v32; // rsi
  __int64 v33; // rcx
  __int8 *v34; // rdi
  __int64 v35; // rsi
  __int64 v36; // rdx
  __m128i v37; // xmm0
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // [rsp+0h] [rbp-E0h]
  __m128i v41; // [rsp+10h] [rbp-D0h] BYREF
  void *dest; // [rsp+20h] [rbp-C0h] BYREF
  __m128i v43; // [rsp+28h] [rbp-B8h] BYREF
  const char *v44; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v45; // [rsp+48h] [rbp-98h]
  _WORD *v46; // [rsp+50h] [rbp-90h]
  __int64 v47; // [rsp+58h] [rbp-88h]
  __int16 v48; // [rsp+60h] [rbp-80h]
  __int8 *v49; // [rsp+70h] [rbp-70h] BYREF
  __m128i src; // [rsp+78h] [rbp-68h] BYREF
  _QWORD *v51; // [rsp+88h] [rbp-58h]
  __int64 v52; // [rsp+90h] [rbp-50h]
  _QWORD v53[9]; // [rsp+98h] [rbp-48h] BYREF

  v3 = *(_QWORD *)(a2 + 64);
  if ( v3 )
  {
    v4 = *(_WORD **)(a2 + 56);
    if ( v3 != 1 || *(_BYTE *)v4 != 33 )
    {
      v43.m128i_i8[8] = 0;
      v43.m128i_i64[0] = 0;
      dest = &v43.m128i_u64[1];
      v5 = v3;
      while ( 1 )
      {
        --v5;
        if ( *((_BYTE *)v4 + v5) == 33 )
          break;
        if ( !v5 )
          goto LABEL_6;
      }
      if ( !v5 )
      {
        v29 = **(_QWORD **)(a2 + 8);
        v45 = 1;
        v44 = (const char *)&unk_3F6A4C5;
        v30 = sub_CA94E0(v29 + 112, (__int64)&v44);
        v31 = *(_QWORD *)(v30 + 56);
        v32 = *(_BYTE **)(v30 + 48);
        v49 = &src.m128i_i8[8];
        sub_CA61F0((__int64 *)&v49, v32, (__int64)&v32[v31]);
        v34 = (__int8 *)dest;
        if ( v49 == (__int8 *)&src.m128i_u64[1] )
        {
          v38 = src.m128i_i64[0];
          if ( src.m128i_i64[0] )
          {
            if ( src.m128i_i64[0] == 1 )
              *(_BYTE *)dest = src.m128i_i8[8];
            else
              memcpy(dest, &src.m128i_u64[1], src.m128i_u64[0]);
            v38 = src.m128i_i64[0];
            v34 = (__int8 *)dest;
          }
          v43.m128i_i64[0] = v38;
          v34[v38] = 0;
          v34 = v49;
        }
        else
        {
          v33 = src.m128i_i64[0];
          if ( dest == &v43.m128i_u64[1] )
          {
            dest = v49;
            v43 = src;
          }
          else
          {
            v35 = v43.m128i_i64[1];
            dest = v49;
            v43 = src;
            if ( v34 )
            {
              v49 = v34;
              src.m128i_i64[1] = v35;
              goto LABEL_50;
            }
          }
          v49 = &src.m128i_i8[8];
          v34 = &src.m128i_i8[8];
        }
LABEL_50:
        src.m128i_i64[0] = 0;
        *v34 = 0;
        if ( v49 != (__int8 *)&src.m128i_u64[1] )
          j_j___libc_free_0(v49, src.m128i_i64[1] + 1);
        if ( 0x3FFFFFFFFFFFFFFFLL - v43.m128i_i64[0] >= v3 - 1 )
        {
          sub_2241490(&dest, (char *)v4 + 1, v3 - 1, v33);
          a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
          v19 = dest;
          if ( dest != &v43.m128i_u64[1] )
            goto LABEL_54;
LABEL_23:
          a1[1] = _mm_load_si128((const __m128i *)&v43.m128i_u64[1]);
LABEL_55:
          a1->m128i_i64[1] = v43.m128i_i64[0];
          return a1;
        }
LABEL_89:
        sub_4262D8((__int64)"basic_string::append");
      }
LABEL_6:
      v6 = **(_QWORD **)(a2 + 8);
      if ( v3 == 1 || *v4 != 8481 )
      {
        v7 = v3;
        while ( 1 )
        {
          v8 = v7--;
          if ( *((_BYTE *)v4 + v7) == 33 )
            break;
          if ( !v7 )
            goto LABEL_11;
        }
        if ( v8 > v3 )
          v8 = v3;
        v7 = v8;
LABEL_11:
        v40 = **(_QWORD **)(a2 + 8);
        v41.m128i_i64[0] = (__int64)v4;
        v41.m128i_i64[1] = v7;
        v9 = sub_CA94E0(v6 + 112, (__int64)&v41);
        if ( v9 == v40 + 120 )
        {
          LOBYTE(v53[0]) = 0;
          v51 = v53;
          v37 = _mm_load_si128(&v41);
          v44 = "Unknown tag handle ";
          v48 = 1283;
          v52 = 0;
          LODWORD(v49) = 22;
          v46 = v4;
          v47 = v7;
          src = v37;
          sub_CA8D00(a2, (__int64)&v44, (__int64)&v49, v10, v11);
          if ( v51 != v53 )
            j_j___libc_free_0(v51, v53[0] + 1LL);
LABEL_18:
          v17 = v3;
          while ( 1 )
          {
            v18 = v17--;
            if ( *((_BYTE *)v4 + v17) == 33 )
              break;
            if ( !v17 )
              goto LABEL_21;
          }
          if ( v3 >= v18 )
          {
            v3 -= v18;
            v4 = (_WORD *)((char *)v4 + v18);
LABEL_21:
            if ( v3 > 0x3FFFFFFFFFFFFFFFLL - v43.m128i_i64[0] )
              goto LABEL_89;
          }
          else
          {
            v4 = (_WORD *)((char *)v4 + v3);
            v3 = 0;
          }
          sub_2241490(&dest, v4, v3, v14);
          a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
          v19 = dest;
          if ( dest == &v43.m128i_u64[1] )
            goto LABEL_23;
LABEL_54:
          a1->m128i_i64[0] = (__int64)v19;
          a1[1].m128i_i64[0] = v43.m128i_i64[1];
          goto LABEL_55;
        }
        v12 = *(_QWORD *)(v9 + 56);
        v13 = *(_BYTE **)(v9 + 48);
        v49 = &src.m128i_i8[8];
        sub_CA61F0((__int64 *)&v49, v13, (__int64)&v13[v12]);
        v15 = (__int8 *)dest;
        if ( v49 == (__int8 *)&src.m128i_u64[1] )
        {
          v39 = src.m128i_i64[0];
          if ( src.m128i_i64[0] )
          {
            if ( src.m128i_i64[0] == 1 )
              *(_BYTE *)dest = src.m128i_i8[8];
            else
              memcpy(dest, &src.m128i_u64[1], src.m128i_u64[0]);
            v39 = src.m128i_i64[0];
            v15 = (__int8 *)dest;
          }
          v43.m128i_i64[0] = v39;
          v15[v39] = 0;
          v15 = v49;
          goto LABEL_16;
        }
        v14 = src.m128i_i64[0];
        if ( dest == &v43.m128i_u64[1] )
        {
          dest = v49;
          v43 = src;
        }
        else
        {
          v16 = v43.m128i_i64[1];
          dest = v49;
          v43 = src;
          if ( v15 )
          {
            v49 = v15;
            src.m128i_i64[1] = v16;
            goto LABEL_16;
          }
        }
        v49 = &src.m128i_i8[8];
        v15 = &src.m128i_i8[8];
LABEL_16:
        src.m128i_i64[0] = 0;
        *v15 = 0;
        if ( v49 != (__int8 *)&src.m128i_u64[1] )
          j_j___libc_free_0(v49, src.m128i_i64[1] + 1);
        goto LABEL_18;
      }
      v45 = 2;
      v44 = (const char *)&unk_3F6A4C4;
      v23 = sub_CA94E0(v6 + 112, (__int64)&v44);
      v24 = *(_QWORD *)(v23 + 56);
      v25 = *(_BYTE **)(v23 + 48);
      v49 = &src.m128i_i8[8];
      sub_CA61F0((__int64 *)&v49, v25, (__int64)&v25[v24]);
      v27 = (__int8 *)dest;
      if ( v49 == (__int8 *)&src.m128i_u64[1] )
      {
        v36 = src.m128i_i64[0];
        if ( src.m128i_i64[0] )
        {
          if ( src.m128i_i64[0] == 1 )
            *(_BYTE *)dest = src.m128i_i8[8];
          else
            memcpy(dest, &src.m128i_u64[1], src.m128i_u64[0]);
          v36 = src.m128i_i64[0];
          v27 = (__int8 *)dest;
        }
        v43.m128i_i64[0] = v36;
        v27[v36] = 0;
        v27 = v49;
        goto LABEL_39;
      }
      v26 = src.m128i_i64[0];
      if ( dest == &v43.m128i_u64[1] )
      {
        dest = v49;
        v43 = src;
      }
      else
      {
        v28 = v43.m128i_i64[1];
        dest = v49;
        v43 = src;
        if ( v27 )
        {
          v49 = v27;
          src.m128i_i64[1] = v28;
LABEL_39:
          src.m128i_i64[0] = 0;
          *v27 = 0;
          if ( v49 != (__int8 *)&src.m128i_u64[1] )
            j_j___libc_free_0(v49, src.m128i_i64[1] + 1);
          if ( 0x3FFFFFFFFFFFFFFFLL - v43.m128i_i64[0] < v3 - 2 )
            goto LABEL_89;
          sub_2241490(&dest, v4 + 1, v3 - 2, v26);
          a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
          v19 = dest;
          if ( dest == &v43.m128i_u64[1] )
            goto LABEL_23;
          goto LABEL_54;
        }
      }
      v49 = &src.m128i_i8[8];
      v27 = &src.m128i_i8[8];
      goto LABEL_39;
    }
  }
  v20 = *(_DWORD *)(a2 + 32);
  v21 = a1 + 1;
  if ( v20 == 4 )
  {
    a1->m128i_i64[0] = (__int64)v21;
    sub_CA61F0(a1->m128i_i64, "tag:yaml.org,2002:map", (__int64)"");
    return a1;
  }
  if ( v20 > 4 )
  {
    if ( v20 == 5 )
    {
      a1->m128i_i64[0] = (__int64)v21;
      sub_CA61F0(a1->m128i_i64, "tag:yaml.org,2002:seq", (__int64)"");
      return a1;
    }
LABEL_34:
    a1->m128i_i64[0] = (__int64)v21;
    sub_CA61F0(a1->m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
    return a1;
  }
  if ( !v20 )
  {
    a1->m128i_i64[0] = (__int64)v21;
    sub_CA61F0(a1->m128i_i64, "tag:yaml.org,2002:null", (__int64)"");
    return a1;
  }
  if ( v20 == 3 )
    goto LABEL_34;
  a1->m128i_i64[0] = (__int64)v21;
  sub_CA61F0(a1->m128i_i64, "tag:yaml.org,2002:str", (__int64)"");
  return a1;
}
