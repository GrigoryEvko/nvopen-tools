// Function: sub_C86E60
// Address: 0xc86e60
//
__m128i *__fastcall sub_C86E60(char *a1, size_t a2, size_t a3, _BYTE *a4, __int64 a5)
{
  __m128i *v6; // r13
  __int64 v7; // r12
  __int64 v8; // rbx
  char *v9; // rax
  _BYTE *v11; // rax
  __int64 v13; // r12
  __int64 v14; // r12
  __int64 v15; // rax
  size_t v16; // r14
  const void *v17; // r8
  char *v18; // rdi
  char *v19; // rax
  char *v20; // r12
  __int64 v21; // rax
  char *v22; // rdi
  const void *v23; // [rsp+10h] [rbp-2C0h]
  _BYTE *v25; // [rsp+20h] [rbp-2B0h]
  __int64 v26; // [rsp+28h] [rbp-2A8h]
  _QWORD v27[4]; // [rsp+30h] [rbp-2A0h] BYREF
  __int16 v28; // [rsp+50h] [rbp-280h]
  char v29[32]; // [rsp+60h] [rbp-270h] BYREF
  __int16 v30; // [rsp+80h] [rbp-250h]
  char v31[32]; // [rsp+90h] [rbp-240h] BYREF
  __int16 v32; // [rsp+B0h] [rbp-220h]
  __int64 v33[2]; // [rsp+C0h] [rbp-210h] BYREF
  __m128i v34; // [rsp+D0h] [rbp-200h] BYREF
  __int16 v35; // [rsp+E0h] [rbp-1F0h]
  char *v36; // [rsp+F0h] [rbp-1E0h] BYREF
  __int64 v37; // [rsp+F8h] [rbp-1D8h]
  unsigned __int64 v38; // [rsp+100h] [rbp-1D0h]
  _BYTE dest[136]; // [rsp+108h] [rbp-1C8h] BYREF
  __m128i *v40; // [rsp+190h] [rbp-140h] BYREF
  __int64 v41; // [rsp+198h] [rbp-138h]
  __m128i v42[19]; // [rsp+1A0h] [rbp-130h] BYREF

  v6 = (__m128i *)a1;
  v7 = a5;
  v8 = (__int64)a4;
  v25 = (_BYTE *)a2;
  if ( a3 )
  {
    v9 = (char *)a2;
    a2 = 47;
    a1 = v9;
    v11 = memchr(v9, 47, a3);
    if ( v11 )
    {
      a4 = v25;
      if ( v11 - v25 != -1 )
      {
        v40 = v42;
        sub_C865D0((__int64 *)&v40, v25, (__int64)&v25[a3]);
        v6[2].m128i_i8[0] &= ~1u;
        v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
        if ( v40 == v42 )
        {
          v6[1] = _mm_load_si128(v42);
        }
        else
        {
          v6->m128i_i64[0] = (__int64)v40;
          v6[1].m128i_i64[0] = v42[0].m128i_i64[0];
        }
        v6->m128i_i64[1] = v41;
        return v6;
      }
    }
  }
  v40 = v42;
  v41 = 0x1000000000LL;
  if ( !a5 )
  {
    a1 = "PATH";
    v19 = getenv("PATH");
    v20 = v19;
    if ( !v19 )
      goto LABEL_23;
    a1 = v19;
    a2 = strlen(v19);
    sub_C92330(v20, a2, &v40, ":", 1);
    v8 = (__int64)v40;
    v7 = (unsigned int)v41;
  }
  v13 = 16 * v7;
  v26 = v8 + v13;
  if ( v8 + v13 == v8 )
  {
LABEL_23:
    v6[2].m128i_i8[0] |= 1u;
    v21 = sub_2241E50(a1, a2, a3, a4, a5);
    v6->m128i_i32[0] = 2;
    v6->m128i_i64[1] = v21;
    goto LABEL_24;
  }
  v14 = v8;
  while ( 1 )
  {
    v16 = *(_QWORD *)(v14 + 8);
    if ( v16 )
      break;
LABEL_18:
    v14 += 16;
    if ( v26 == v14 )
      goto LABEL_23;
  }
  v17 = *(const void **)v14;
  v37 = 0;
  v36 = dest;
  v18 = dest;
  v38 = 128;
  if ( v16 > 0x80 )
  {
    v23 = v17;
    sub_C8D290(&v36, dest, v16, 1);
    v17 = v23;
    v18 = &v36[v37];
  }
  memcpy(v18, v17, v16);
  v37 += v16;
  v27[0] = v25;
  v35 = 257;
  v32 = 257;
  a2 = (size_t)v27;
  v30 = 257;
  v28 = 261;
  v27[1] = a3;
  sub_C81B70(&v36, (__int64)v27, (__int64)v29, (__int64)v31, (__int64)v33);
  v15 = v37;
  if ( v37 + 1 > v38 )
  {
    a2 = (size_t)dest;
    sub_C8D290(&v36, dest, v37 + 1, 1);
    v15 = v37;
  }
  v36[v15] = 0;
  v35 = 257;
  if ( *v36 )
  {
    v33[0] = (__int64)v36;
    LOBYTE(v35) = 3;
  }
  if ( !sub_C826C0((__int64)v33) )
  {
    a1 = v36;
    if ( v36 != dest )
      _libc_free(v36, a2);
    goto LABEL_18;
  }
  a2 = (size_t)v36;
  v33[0] = (__int64)&v34;
  sub_C865D0(v33, v36, (__int64)&v36[v37]);
  v6[2].m128i_i8[0] &= ~1u;
  v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
  if ( (__m128i *)v33[0] == &v34 )
  {
    v6[1] = _mm_load_si128(&v34);
  }
  else
  {
    v6->m128i_i64[0] = v33[0];
    v6[1].m128i_i64[0] = v34.m128i_i64[0];
  }
  v22 = v36;
  v6->m128i_i64[1] = v33[1];
  if ( v22 != dest )
    _libc_free(v22, a2);
LABEL_24:
  if ( v40 != v42 )
    _libc_free(v40, a2);
  return v6;
}
