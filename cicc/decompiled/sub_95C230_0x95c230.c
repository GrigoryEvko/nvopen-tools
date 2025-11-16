// Function: sub_95C230
// Address: 0x95c230
//
__int64 __fastcall sub_95C230(const char *s1, __int64 *a2, _DWORD *a3)
{
  size_t v4; // r12
  size_t v5; // rdx
  _QWORD *v6; // r8
  int v7; // eax
  int v8; // eax
  _QWORD *v9; // r8
  _QWORD *v11; // r8
  size_t v12; // rdx
  int v13; // eax
  _QWORD *v14; // r8
  __m128i *v15; // rax
  __int64 v16; // rcx
  __m128i *v17; // rax
  __int64 v18; // rcx
  __m128i *v19; // r10
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rcx
  __m128i *v23; // rax
  __m128i *v24; // rcx
  __m128i *v25; // rdx
  __int64 v26; // rbx
  __int64 v27; // rax
  _QWORD *v28; // [rsp+0h] [rbp-F0h]
  int v29; // [rsp+0h] [rbp-F0h]
  void *s1b; // [rsp+8h] [rbp-E8h]
  char *s1a; // [rsp+8h] [rbp-E8h]
  _QWORD *v32; // [rsp+10h] [rbp-E0h]
  size_t v33; // [rsp+10h] [rbp-E0h]
  size_t v34; // [rsp+10h] [rbp-E0h]
  _QWORD *v35; // [rsp+10h] [rbp-E0h]
  int v36; // [rsp+10h] [rbp-E0h]
  __m128i *v37; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v38; // [rsp+28h] [rbp-C8h]
  __m128i v39; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v40[2]; // [rsp+40h] [rbp-B0h] BYREF
  _QWORD v41[2]; // [rsp+50h] [rbp-A0h] BYREF
  __m128i *v42; // [rsp+60h] [rbp-90h] BYREF
  __int64 v43; // [rsp+68h] [rbp-88h]
  __m128i v44; // [rsp+70h] [rbp-80h] BYREF
  __m128i *v45; // [rsp+80h] [rbp-70h] BYREF
  __int64 v46; // [rsp+88h] [rbp-68h]
  __m128i v47; // [rsp+90h] [rbp-60h] BYREF
  void *s2; // [rsp+A0h] [rbp-50h] BYREF
  size_t n; // [rsp+A8h] [rbp-48h]
  _QWORD v50[8]; // [rsp+B0h] [rbp-40h] BYREF

  v4 = 0;
  if ( s1 )
    v4 = strlen(s1);
  s2 = v50;
  sub_95BB50((__int64 *)&s2, (char)&byte_3F157F6[-14], &byte_3F157F6[-14], byte_3F157F6);
  v5 = n;
  v6 = s2;
  if ( n <= v4 )
  {
    if ( !n || (s1b = (void *)n, v32 = s2, v7 = memcmp(s1, s2, n), v6 = v32, v5 = (size_t)s1b, !v7) )
    {
      if ( v6 != v50 )
      {
        v33 = v5;
        j_j___libc_free_0(v6, v50[0] + 1LL);
        v5 = v33;
      }
      s2 = v50;
      s1a = (char *)&s1[v5];
      v34 = v4 - v5;
      sub_95BB50((__int64 *)&s2, (char)&byte_3F157E2[-6], &byte_3F157E2[-6], byte_3F157E2);
      if ( n == v34 )
      {
        if ( !v34 )
        {
          if ( s2 != v50 )
            j_j___libc_free_0(s2, v50[0] + 1LL);
          goto LABEL_13;
        }
        v28 = s2;
        v8 = memcmp(s1a, s2, v34);
        v9 = v28;
        if ( v28 != v50 )
        {
          v29 = v8;
          j_j___libc_free_0(v9, v50[0] + 1LL);
          v8 = v29;
        }
        if ( !v8 )
        {
LABEL_13:
          *a3 |= 0x100u;
          return 1;
        }
        s2 = v50;
        sub_95BB50((__int64 *)&s2, (char)&byte_3F157DB[-11], &byte_3F157DB[-11], byte_3F157DB);
        v11 = s2;
        if ( v34 != n )
        {
LABEL_27:
          if ( v11 != v50 )
            j_j___libc_free_0(v11, v50[0] + 1LL);
          if ( !a2 )
            return 0xFFFFFFFFLL;
          s2 = v50;
          sub_95BB50((__int64 *)&s2, (char)&byte_3F157CD[-13], &byte_3F157CD[-13], byte_3F157CD);
          if ( !s1a )
          {
            LOBYTE(v41[0]) = 0;
            v40[0] = (__int64)v41;
            v40[1] = 0;
            goto LABEL_32;
          }
LABEL_62:
          v40[0] = (__int64)v41;
          sub_95BA30(v40, s1a, (__int64)&s1[v4]);
LABEL_32:
          v15 = (__m128i *)sub_2241130(v40, 0, 0, "libnvvm : error: ", 17);
          v42 = &v44;
          if ( (__m128i *)v15->m128i_i64[0] == &v15[1] )
          {
            v44 = _mm_loadu_si128(v15 + 1);
          }
          else
          {
            v42 = (__m128i *)v15->m128i_i64[0];
            v44.m128i_i64[0] = v15[1].m128i_i64[0];
          }
          v16 = v15->m128i_i64[1];
          v15[1].m128i_i8[0] = 0;
          v43 = v16;
          v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
          v15->m128i_i64[1] = 0;
          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v43) <= 0x24 )
            sub_4262D8((__int64)"basic_string::append");
          v17 = (__m128i *)sub_2241490(&v42, " is an unsupported value for option: ", 37, v16);
          v45 = &v47;
          if ( (__m128i *)v17->m128i_i64[0] == &v17[1] )
          {
            v47 = _mm_loadu_si128(v17 + 1);
          }
          else
          {
            v45 = (__m128i *)v17->m128i_i64[0];
            v47.m128i_i64[0] = v17[1].m128i_i64[0];
          }
          v18 = v17->m128i_i64[1];
          v17[1].m128i_i8[0] = 0;
          v46 = v18;
          v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
          v19 = v45;
          v17->m128i_i64[1] = 0;
          v20 = 15;
          v21 = 15;
          if ( v19 != &v47 )
            v21 = v47.m128i_i64[0];
          v22 = v46 + n;
          if ( v46 + n <= v21 )
            goto LABEL_43;
          if ( s2 != v50 )
            v20 = v50[0];
          if ( v22 <= v20 )
          {
            v23 = (__m128i *)sub_2241130(&s2, 0, 0, v19, v46);
            v37 = &v39;
            v24 = (__m128i *)v23->m128i_i64[0];
            v25 = v23 + 1;
            if ( (__m128i *)v23->m128i_i64[0] != &v23[1] )
              goto LABEL_44;
          }
          else
          {
LABEL_43:
            v23 = (__m128i *)sub_2241490(&v45, s2, n, v22);
            v37 = &v39;
            v24 = (__m128i *)v23->m128i_i64[0];
            v25 = v23 + 1;
            if ( (__m128i *)v23->m128i_i64[0] != &v23[1] )
            {
LABEL_44:
              v37 = v24;
              v39.m128i_i64[0] = v23[1].m128i_i64[0];
LABEL_45:
              v38 = v23->m128i_i64[1];
              v23->m128i_i64[0] = (__int64)v25;
              v23->m128i_i64[1] = 0;
              v23[1].m128i_i8[0] = 0;
              if ( v45 != &v47 )
                j_j___libc_free_0(v45, v47.m128i_i64[0] + 1);
              if ( v42 != &v44 )
                j_j___libc_free_0(v42, v44.m128i_i64[0] + 1);
              if ( (_QWORD *)v40[0] != v41 )
                j_j___libc_free_0(v40[0], v41[0] + 1LL);
              if ( s2 != v50 )
                j_j___libc_free_0(s2, v50[0] + 1LL);
              v26 = v38;
              v27 = sub_2207820(v38 + 1);
              *a2 = v27;
              sub_2241570(&v37, v27, v26, 0);
              *(_BYTE *)(*a2 + v26) = 0;
              if ( v37 != &v39 )
                j_j___libc_free_0(v37, v39.m128i_i64[0] + 1);
              return 0xFFFFFFFFLL;
            }
          }
          v39 = _mm_loadu_si128(v23 + 1);
          goto LABEL_45;
        }
      }
      else
      {
        if ( s2 != v50 )
          j_j___libc_free_0(s2, v50[0] + 1LL);
        s2 = v50;
        sub_95BB50((__int64 *)&s2, (char)&byte_3F157DB[-11], &byte_3F157DB[-11], byte_3F157DB);
        v11 = s2;
        if ( n != v34 )
          goto LABEL_27;
        if ( !v34 )
        {
          if ( s2 != v50 )
            j_j___libc_free_0(s2, v50[0] + 1LL);
          goto LABEL_25;
        }
      }
      v12 = v34;
      v35 = v11;
      v13 = memcmp(s1a, v11, v12);
      v14 = v35;
      if ( v35 != v50 )
      {
        v36 = v13;
        j_j___libc_free_0(v14, v50[0] + 1LL);
        v13 = v36;
      }
      if ( v13 )
      {
        if ( !a2 )
          return 0xFFFFFFFFLL;
        s2 = v50;
        sub_95BB50((__int64 *)&s2, (char)&byte_3F157CD[-13], &byte_3F157CD[-13], byte_3F157CD);
        goto LABEL_62;
      }
LABEL_25:
      *a3 |= 0x200u;
      return 1;
    }
  }
  if ( v6 != v50 )
    j_j___libc_free_0(v6, v50[0] + 1LL);
  return 0;
}
