// Function: sub_C94E30
// Address: 0xc94e30
//
void __fastcall __noreturn sub_C94E30(const char *a1, unsigned int a2)
{
  unsigned int v2; // r15d
  size_t v3; // rax
  const char *v4; // r8
  size_t v5; // r14
  __int64 v6; // rcx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rcx
  __m128i *v10; // rax
  __m128i *v11; // rcx
  __m128i *v12; // rdx
  __m128i *v13; // rdi
  size_t v14; // rdx
  __int64 v15; // r8
  _QWORD *v16; // rdi
  __int64 v17; // rax
  void *dest; // [rsp+10h] [rbp-E0h] BYREF
  size_t v19; // [rsp+18h] [rbp-D8h]
  _QWORD v20[2]; // [rsp+20h] [rbp-D0h] BYREF
  _QWORD v21[2]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v22[2]; // [rsp+40h] [rbp-B0h] BYREF
  __m128i *v23; // [rsp+50h] [rbp-A0h] BYREF
  size_t n; // [rsp+58h] [rbp-98h]
  __m128i v25; // [rsp+60h] [rbp-90h] BYREF
  unsigned __int64 *v26; // [rsp+70h] [rbp-80h] BYREF
  __int64 v27; // [rsp+78h] [rbp-78h]
  unsigned __int64 v28; // [rsp+80h] [rbp-70h] BYREF
  void **p_dest; // [rsp+90h] [rbp-60h] BYREF
  __int64 v30; // [rsp+98h] [rbp-58h]
  unsigned __int64 v31; // [rsp+A0h] [rbp-50h] BYREF
  __int16 v32; // [rsp+B0h] [rbp-40h]

  v2 = a2;
  v21[0] = v22;
  dest = v20;
  v19 = 0;
  LOBYTE(v20[0]) = 0;
  v3 = strlen(a1);
  v4 = a1;
  p_dest = (void **)v3;
  v5 = v3;
  if ( v3 > 0xF )
  {
    v17 = sub_22409D0(v21, &p_dest, 0);
    v4 = a1;
    v21[0] = v17;
    v16 = (_QWORD *)v17;
    v22[0] = p_dest;
  }
  else
  {
    if ( v3 == 1 )
    {
      LOBYTE(v22[0]) = *a1;
      goto LABEL_4;
    }
    if ( !v3 )
    {
LABEL_4:
      v21[1] = p_dest;
      *((_BYTE *)p_dest + v21[0]) = 0;
      if ( a2 == -1 )
        v2 = *__errno_location();
      sub_F03820(&v26, v2);
      sub_2241BD0(&p_dest, v21);
      if ( v30 == 0x3FFFFFFFFFFFFFFFLL || v30 == 4611686018427387902LL )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490(&p_dest, ": ", 2, v6);
      v7 = 15;
      v8 = 15;
      if ( p_dest != (void **)&v31 )
        v8 = v31;
      v9 = v30 + v27;
      if ( v30 + v27 <= v8 )
        goto LABEL_13;
      if ( v26 != &v28 )
        v7 = v28;
      if ( v9 <= v7 )
      {
        v10 = (__m128i *)sub_2241130(&v26, 0, 0, p_dest, v30);
        v23 = &v25;
        v11 = (__m128i *)v10->m128i_i64[0];
        v12 = v10 + 1;
        if ( (__m128i *)v10->m128i_i64[0] != &v10[1] )
          goto LABEL_14;
      }
      else
      {
LABEL_13:
        v10 = (__m128i *)sub_2241490(&p_dest, v26, v27, v9);
        v23 = &v25;
        v11 = (__m128i *)v10->m128i_i64[0];
        v12 = v10 + 1;
        if ( (__m128i *)v10->m128i_i64[0] != &v10[1] )
        {
LABEL_14:
          v23 = v11;
          v25.m128i_i64[0] = v10[1].m128i_i64[0];
          goto LABEL_15;
        }
      }
      v25 = _mm_loadu_si128(v10 + 1);
LABEL_15:
      n = v10->m128i_u64[1];
      v10->m128i_i64[0] = (__int64)v12;
      v10->m128i_i64[1] = 0;
      v10[1].m128i_i8[0] = 0;
      v13 = (__m128i *)dest;
      v14 = n;
      if ( v23 == &v25 )
      {
        if ( n )
        {
          if ( n == 1 )
            *(_BYTE *)dest = v25.m128i_i8[0];
          else
            memcpy(dest, &v25, n);
          v13 = (__m128i *)dest;
          v14 = n;
        }
        v19 = v14;
        v13->m128i_i8[v14] = 0;
      }
      else
      {
        if ( dest == v20 )
        {
          dest = v23;
          v19 = n;
          v20[0] = v25.m128i_i64[0];
        }
        else
        {
          v15 = v20[0];
          dest = v23;
          v19 = n;
          v20[0] = v25.m128i_i64[0];
          if ( v13 )
          {
            v23 = v13;
            v25.m128i_i64[0] = v15;
            goto LABEL_19;
          }
        }
        v23 = &v25;
      }
LABEL_19:
      n = 0;
      v23->m128i_i8[0] = 0;
      sub_2240A30(&v23);
      sub_2240A30(&p_dest);
      sub_2240A30(&v26);
      sub_2240A30(v21);
      v32 = 260;
      p_dest = &dest;
      sub_C64D30((__int64)&p_dest, 1u);
    }
    v16 = v22;
  }
  memcpy(v16, v4, v5);
  goto LABEL_4;
}
