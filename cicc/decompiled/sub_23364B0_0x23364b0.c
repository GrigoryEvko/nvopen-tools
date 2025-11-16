// Function: sub_23364B0
// Address: 0x23364b0
//
__m128i *__fastcall sub_23364B0(__m128i *a1, __int64 a2, size_t a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rcx
  __int64 v6; // rsi
  unsigned __int64 v7; // rdi
  __int64 v8; // rcx
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rax
  char *v14; // rdi
  size_t v15; // rdx
  __int64 v16; // rcx
  __m128i *v17; // rcx
  __int64 v18; // rax
  unsigned int v19; // [rsp+8h] [rbp-E8h]
  __int64 v20; // [rsp+10h] [rbp-E0h] BYREF
  unsigned __int64 v21; // [rsp+18h] [rbp-D8h]
  __int64 v22; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v23; // [rsp+30h] [rbp-C0h] BYREF
  unsigned __int64 v24; // [rsp+38h] [rbp-B8h]
  void *dest; // [rsp+40h] [rbp-B0h]
  size_t v26; // [rsp+48h] [rbp-A8h]
  __m128i v27; // [rsp+50h] [rbp-A0h] BYREF
  unsigned __int64 v28[4]; // [rsp+60h] [rbp-90h] BYREF
  char *v29; // [rsp+80h] [rbp-70h] BYREF
  size_t n; // [rsp+88h] [rbp-68h]
  _QWORD src[2]; // [rsp+90h] [rbp-60h] BYREF
  char v32; // [rsp+A0h] [rbp-50h]
  _QWORD v33[2]; // [rsp+A8h] [rbp-48h] BYREF
  _QWORD *v34; // [rsp+B8h] [rbp-38h] BYREF

  v20 = a2;
  v21 = a3;
  dest = &v27;
  v26 = 0;
  v27.m128i_i8[0] = 0;
  if ( !a3 )
  {
    a1[2].m128i_i8[0] = a1[2].m128i_i8[0] & 0xFC | 2;
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
LABEL_34:
    a1[1] = _mm_load_si128(&v27);
    goto LABEL_24;
  }
  do
  {
    v23 = 0;
    v24 = 0;
    LOBYTE(v29) = 59;
    v4 = sub_C931B0(&v20, &v29, 1u, 0);
    if ( v4 == -1 )
    {
      v6 = v20;
      v4 = v21;
      v7 = 0;
      v8 = 0;
    }
    else
    {
      v5 = v4 + 1;
      v6 = v20;
      if ( v4 + 1 > v21 )
      {
        v5 = v21;
        v7 = 0;
      }
      else
      {
        v7 = v21 - v5;
      }
      v8 = v20 + v5;
      if ( v4 > v21 )
        v4 = v21;
    }
    v23 = v6;
    v24 = v4;
    v20 = v8;
    v21 = v7;
    if ( v4 <= 0x10
      || *(_QWORD *)v6 ^ 0x2D656C69666F7270LL | *(_QWORD *)(v6 + 8) ^ 0x656D616E656C6966LL
      || *(_BYTE *)(v6 + 16) != 61 )
    {
      v9 = sub_C63BB0();
      n = 40;
      v11 = v10;
      v32 = 1;
      v19 = v9;
      v29 = "invalid MemProfUse pass parameter '{0}' ";
      src[0] = &v34;
      src[1] = 1;
      v33[0] = &unk_49DB108;
      v33[1] = &v23;
      v34 = v33;
      sub_23328D0((__int64)v28, (__int64)&v29);
      sub_23058C0(&v22, (__int64)v28, v19, v11);
      v12 = v22;
      a1[2].m128i_i8[0] |= 3u;
      a1->m128i_i64[0] = v12 & 0xFFFFFFFFFFFFFFFELL;
      sub_2240A30(v28);
      if ( dest != &v27 )
        j_j___libc_free_0((unsigned __int64)dest);
      return a1;
    }
    v29 = (char *)src;
    v24 = v4 - 17;
    v23 = v6 + 17;
    sub_2305260((__int64 *)&v29, (_BYTE *)(v6 + 17), v6 + v4);
    v14 = (char *)dest;
    v15 = n;
    if ( v29 == (char *)src )
    {
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = src[0];
        else
          memcpy(dest, src, n);
        v15 = n;
        v14 = (char *)dest;
      }
      v26 = v15;
      v14[v15] = 0;
      v14 = v29;
    }
    else
    {
      if ( dest == &v27 )
      {
        dest = v29;
        v26 = n;
        v27.m128i_i64[0] = src[0];
      }
      else
      {
        v16 = v27.m128i_i64[0];
        dest = v29;
        v26 = n;
        v27.m128i_i64[0] = src[0];
        if ( v14 )
        {
          v29 = v14;
          src[0] = v16;
          goto LABEL_19;
        }
      }
      v29 = (char *)src;
      v14 = (char *)src;
    }
LABEL_19:
    n = 0;
    *v14 = 0;
    if ( v29 != (char *)src )
      j_j___libc_free_0((unsigned __int64)v29);
  }
  while ( v21 );
  v17 = (__m128i *)dest;
  a3 = v26;
  a1[2].m128i_i8[0] = a1[2].m128i_i8[0] & 0xFC | 2;
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( v17 == &v27 )
    goto LABEL_34;
  v18 = v27.m128i_i64[0];
  a1->m128i_i64[0] = (__int64)v17;
  a1[1].m128i_i64[0] = v18;
LABEL_24:
  a1->m128i_i64[1] = a3;
  return a1;
}
