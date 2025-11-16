// Function: sub_12C6910
// Address: 0x12c6910
//
__int64 __fastcall sub_12C6910(const char *s1, __int64 *a2, _DWORD *a3)
{
  size_t v4; // r12
  size_t v5; // rdx
  _QWORD *v6; // r8
  int v7; // eax
  _QWORD *v8; // r8
  _QWORD *v9; // r8
  __m128i *v10; // rax
  __int64 v11; // rcx
  __m128i *v12; // rax
  __int64 v13; // rcx
  __m128i *v14; // r10
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rcx
  __m128i *v18; // rax
  __m128i *v19; // rcx
  __m128i *v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // rax
  int v24; // eax
  size_t v25; // rdx
  int v26; // eax
  _QWORD *v27; // [rsp+0h] [rbp-F0h]
  void *s1b; // [rsp+8h] [rbp-E8h]
  char *s1a; // [rsp+8h] [rbp-E8h]
  _QWORD *v30; // [rsp+10h] [rbp-E0h]
  size_t v31; // [rsp+10h] [rbp-E0h]
  _QWORD *v32; // [rsp+10h] [rbp-E0h]
  __m128i *v33; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v34; // [rsp+28h] [rbp-C8h]
  __m128i v35; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v36[2]; // [rsp+40h] [rbp-B0h] BYREF
  _QWORD v37[2]; // [rsp+50h] [rbp-A0h] BYREF
  __m128i *v38; // [rsp+60h] [rbp-90h] BYREF
  __int64 v39; // [rsp+68h] [rbp-88h]
  __m128i v40; // [rsp+70h] [rbp-80h] BYREF
  __m128i *v41; // [rsp+80h] [rbp-70h] BYREF
  __int64 v42; // [rsp+88h] [rbp-68h]
  __m128i v43; // [rsp+90h] [rbp-60h] BYREF
  void *s2; // [rsp+A0h] [rbp-50h] BYREF
  size_t n; // [rsp+A8h] [rbp-48h]
  _QWORD v46[8]; // [rsp+B0h] [rbp-40h] BYREF

  v4 = 0;
  if ( s1 )
    v4 = strlen(s1);
  s2 = v46;
  sub_12C6230((__int64 *)&s2, (char)&byte_42819FE[-14], &byte_42819FE[-14], byte_42819FE);
  v5 = n;
  v6 = s2;
  if ( v4 >= n )
  {
    if ( !n || (s1b = (void *)n, v30 = s2, v7 = memcmp(s1, s2, n), v6 = v30, v5 = (size_t)s1b, !v7) )
    {
      v31 = v4 - v5;
      s1a = (char *)&s1[v5];
      if ( v6 != v46 )
        j_j___libc_free_0(v6, v46[0] + 1LL);
      s2 = v46;
      sub_12C6230((__int64 *)&s2, (char)&byte_42819EA[-6], &byte_42819EA[-6], byte_42819EA);
      v8 = s2;
      if ( v31 == n )
      {
        if ( !v31 || (v27 = s2, v24 = memcmp(s1a, s2, v31), v8 = v27, !v24) )
        {
          if ( v8 != v46 )
            j_j___libc_free_0(v8, v46[0] + 1LL);
          *a3 |= 0x100u;
          return 1;
        }
      }
      if ( v8 != v46 )
        j_j___libc_free_0(v8, v46[0] + 1LL);
      s2 = v46;
      sub_12C6230((__int64 *)&s2, (char)&byte_42819E3[-11], &byte_42819E3[-11], byte_42819E3);
      v9 = s2;
      if ( v31 == n )
      {
        if ( !v31 || (v25 = v31, v32 = s2, v26 = memcmp(s1a, s2, v25), v9 = v32, !v26) )
        {
          if ( v9 != v46 )
            j_j___libc_free_0(v9, v46[0] + 1LL);
          *a3 |= 0x200u;
          return 1;
        }
      }
      if ( v9 != v46 )
        j_j___libc_free_0(v9, v46[0] + 1LL);
      if ( !a2 )
        return 0xFFFFFFFFLL;
      s2 = v46;
      sub_12C6230((__int64 *)&s2, (char)&byte_42819D5[-13], &byte_42819D5[-13], byte_42819D5);
      if ( s1a )
      {
        v36[0] = (__int64)v37;
        sub_12C6150(v36, s1a, (__int64)&s1[v4]);
      }
      else
      {
        LOBYTE(v37[0]) = 0;
        v36[0] = (__int64)v37;
        v36[1] = 0;
      }
      v10 = (__m128i *)sub_2241130(v36, 0, 0, "libnvvm : error: ", 17);
      v38 = &v40;
      if ( (__m128i *)v10->m128i_i64[0] == &v10[1] )
      {
        v40 = _mm_loadu_si128(v10 + 1);
      }
      else
      {
        v38 = (__m128i *)v10->m128i_i64[0];
        v40.m128i_i64[0] = v10[1].m128i_i64[0];
      }
      v11 = v10->m128i_i64[1];
      v10[1].m128i_i8[0] = 0;
      v39 = v11;
      v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
      v10->m128i_i64[1] = 0;
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v39) <= 0x24 )
        sub_4262D8((__int64)"basic_string::append");
      v12 = (__m128i *)sub_2241490(&v38, " is an unsupported value for option: ", 37, v11);
      v41 = &v43;
      if ( (__m128i *)v12->m128i_i64[0] == &v12[1] )
      {
        v43 = _mm_loadu_si128(v12 + 1);
      }
      else
      {
        v41 = (__m128i *)v12->m128i_i64[0];
        v43.m128i_i64[0] = v12[1].m128i_i64[0];
      }
      v13 = v12->m128i_i64[1];
      v12[1].m128i_i8[0] = 0;
      v42 = v13;
      v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
      v14 = v41;
      v12->m128i_i64[1] = 0;
      v15 = 15;
      v16 = 15;
      if ( v14 != &v43 )
        v16 = v43.m128i_i64[0];
      v17 = v42 + n;
      if ( v42 + n <= v16 )
        goto LABEL_28;
      if ( s2 != v46 )
        v15 = v46[0];
      if ( v17 <= v15 )
      {
        v18 = (__m128i *)sub_2241130(&s2, 0, 0, v14, v42);
        v33 = &v35;
        v19 = (__m128i *)v18->m128i_i64[0];
        v20 = v18 + 1;
        if ( (__m128i *)v18->m128i_i64[0] != &v18[1] )
          goto LABEL_29;
      }
      else
      {
LABEL_28:
        v18 = (__m128i *)sub_2241490(&v41, s2, n, v17);
        v33 = &v35;
        v19 = (__m128i *)v18->m128i_i64[0];
        v20 = v18 + 1;
        if ( (__m128i *)v18->m128i_i64[0] != &v18[1] )
        {
LABEL_29:
          v33 = v19;
          v35.m128i_i64[0] = v18[1].m128i_i64[0];
LABEL_30:
          v34 = v18->m128i_i64[1];
          v18->m128i_i64[0] = (__int64)v20;
          v18->m128i_i64[1] = 0;
          v18[1].m128i_i8[0] = 0;
          if ( v41 != &v43 )
            j_j___libc_free_0(v41, v43.m128i_i64[0] + 1);
          if ( v38 != &v40 )
            j_j___libc_free_0(v38, v40.m128i_i64[0] + 1);
          if ( (_QWORD *)v36[0] != v37 )
            j_j___libc_free_0(v36[0], v37[0] + 1LL);
          if ( s2 != v46 )
            j_j___libc_free_0(s2, v46[0] + 1LL);
          v21 = v34;
          v22 = sub_2207820(v34 + 1);
          *a2 = v22;
          sub_2241570(&v33, v22, v21, 0);
          *(_BYTE *)(*a2 + v21) = 0;
          if ( v33 != &v35 )
            j_j___libc_free_0(v33, v35.m128i_i64[0] + 1);
          return 0xFFFFFFFFLL;
        }
      }
      v35 = _mm_loadu_si128(v18 + 1);
      goto LABEL_30;
    }
  }
  if ( v6 != v46 )
    j_j___libc_free_0(v6, v46[0] + 1LL);
  return 0;
}
