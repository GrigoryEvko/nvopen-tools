// Function: sub_16C7620
// Address: 0x16c7620
//
__m128i *__fastcall sub_16C7620(char *a1, size_t a2, _BYTE *a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  bool v7; // sf
  _BYTE *v9; // r14
  __int8 *v10; // r15
  size_t v11; // rdx
  _BYTE *v12; // rax
  __m128i *v13; // rax
  unsigned __int64 v15; // r9
  __int64 v16; // r12
  unsigned __int64 v17; // r12
  __int64 v18; // rax
  size_t v19; // r15
  const void *v20; // r14
  char *v21; // rdi
  char *v22; // rax
  char *v23; // r12
  __int64 v24; // rax
  _BYTE *v25; // r12
  size_t v26; // r13
  __m128i *v27; // rax
  __m128i *v28; // rdx
  __int64 v29; // rax
  __m128i *v30; // rdi
  __m128i *v31; // rdi
  __m128i *v32; // [rsp+10h] [rbp-270h]
  unsigned __int64 v33; // [rsp+18h] [rbp-268h]
  _QWORD v34[2]; // [rsp+20h] [rbp-260h] BYREF
  _QWORD v35[2]; // [rsp+30h] [rbp-250h] BYREF
  __int16 v36; // [rsp+40h] [rbp-240h]
  char v37[16]; // [rsp+50h] [rbp-230h] BYREF
  __int16 v38; // [rsp+60h] [rbp-220h]
  _QWORD v39[2]; // [rsp+70h] [rbp-210h] BYREF
  __int16 v40; // [rsp+80h] [rbp-200h]
  __m128i *v41; // [rsp+90h] [rbp-1F0h] BYREF
  size_t v42; // [rsp+98h] [rbp-1E8h]
  __m128i v43; // [rsp+A0h] [rbp-1E0h] BYREF
  void *src; // [rsp+B0h] [rbp-1D0h] BYREF
  size_t n; // [rsp+B8h] [rbp-1C8h]
  _BYTE dest[128]; // [rsp+C0h] [rbp-1C0h] BYREF
  __m128i *v47; // [rsp+140h] [rbp-140h] BYREF
  __int64 v48; // [rsp+148h] [rbp-138h]
  __m128i v49[19]; // [rsp+150h] [rbp-130h] BYREF

  v5 = a5;
  v32 = (__m128i *)a1;
  v34[0] = a2;
  v34[1] = a3;
  v7 = (__int64)a3 < 0;
  if ( a3 )
  {
    v9 = a3;
    v10 = (__int8 *)a2;
    a2 = 47;
    v11 = 0x7FFFFFFFFFFFFFFFLL;
    a1 = v10;
    if ( !v7 )
      v11 = (size_t)a3;
    v12 = memchr(v10, 47, v11);
    if ( v12 )
    {
      if ( v12 - v10 != -1 )
      {
        src = a3;
        v47 = v49;
        if ( (unsigned __int64)a3 > 0xF )
        {
          v47 = (__m128i *)sub_22409D0(&v47, &src, 0);
          v30 = v47;
          v49[0].m128i_i64[0] = (__int64)src;
        }
        else
        {
          if ( a3 == (_BYTE *)1 )
          {
            v49[0].m128i_i8[0] = *v10;
            v13 = v49;
LABEL_9:
            v48 = (__int64)v9;
            v9[(_QWORD)v13] = 0;
            v32[2].m128i_i8[0] &= ~1u;
            v32->m128i_i64[0] = (__int64)v32[1].m128i_i64;
            if ( v47 == v49 )
            {
              v32[1] = _mm_load_si128(v49);
            }
            else
            {
              v32->m128i_i64[0] = (__int64)v47;
              v32[1].m128i_i64[0] = v49[0].m128i_i64[0];
            }
            v32->m128i_i64[1] = v48;
            return v32;
          }
          v30 = v49;
        }
        memcpy(v30, v10, (size_t)a3);
        v9 = src;
        v13 = v47;
        goto LABEL_9;
      }
    }
  }
  v15 = a4;
  v47 = v49;
  v48 = 0x1000000000LL;
  if ( !v5 )
  {
    a1 = "PATH";
    v22 = getenv("PATH");
    v23 = v22;
    if ( !v22 )
      goto LABEL_28;
    a1 = v22;
    a2 = strlen(v22);
    sub_16D1630(v23, a2, &v47, ":", 1);
    v15 = (unsigned __int64)v47;
    v5 = (unsigned int)v48;
  }
  v16 = 16 * v5;
  v33 = v15 + v16;
  if ( v15 == v15 + v16 )
  {
LABEL_28:
    v32[2].m128i_i8[0] |= 1u;
    v24 = sub_2241E50(a1, a2, a3, a4, a5);
    v32->m128i_i32[0] = 2;
    v32->m128i_i64[1] = v24;
    goto LABEL_29;
  }
  v17 = v15;
  while ( 1 )
  {
    v19 = *(_QWORD *)(v17 + 8);
    if ( v19 )
      break;
LABEL_23:
    v17 += 16LL;
    if ( v33 == v17 )
      goto LABEL_28;
  }
  v20 = *(const void **)v17;
  src = dest;
  v21 = dest;
  n = 0x8000000000LL;
  if ( v19 > 0x80 )
  {
    sub_16CD150(&src, dest, v19, 1);
    v21 = (char *)src + (unsigned int)n;
  }
  memcpy(v21, v20, v19);
  LODWORD(n) = v19 + n;
  v43.m128i_i16[0] = 257;
  v40 = 257;
  v38 = 257;
  a2 = (size_t)v35;
  v36 = 261;
  v35[0] = v34;
  sub_16C4D40((__int64)&src, (__int64)v35, (__int64)v37, (__int64)v39, (__int64)&v41);
  v18 = (unsigned int)n;
  if ( (unsigned int)n >= HIDWORD(n) )
  {
    a2 = (size_t)dest;
    sub_16CD150(&src, dest, 0, 1);
    v18 = (unsigned int)n;
  }
  *((_BYTE *)src + v18) = 0;
  v43.m128i_i16[0] = 257;
  if ( *(_BYTE *)src )
  {
    v41 = (__m128i *)src;
    v43.m128i_i8[0] = 3;
  }
  if ( !sub_16C55D0((__int64)&v41) )
  {
    a1 = (char *)src;
    if ( src != dest )
      _libc_free((unsigned __int64)src);
    goto LABEL_23;
  }
  v25 = src;
  if ( !src )
  {
    v43.m128i_i8[0] = 0;
    v32[2].m128i_i8[0] &= ~1u;
    v32->m128i_i64[0] = (__int64)v32[1].m128i_i64;
    v29 = 0;
LABEL_39:
    v32[1] = _mm_load_si128(&v43);
    goto LABEL_40;
  }
  v26 = (unsigned int)n;
  v41 = &v43;
  v39[0] = (unsigned int)n;
  if ( (unsigned int)n > 0xFuLL )
  {
    v41 = (__m128i *)sub_22409D0(&v41, v39, 0);
    v31 = v41;
    v43.m128i_i64[0] = v39[0];
  }
  else
  {
    if ( (unsigned int)n == 1 )
    {
      v43.m128i_i8[0] = *(_BYTE *)src;
      v27 = &v43;
      goto LABEL_35;
    }
    if ( !(_DWORD)n )
    {
      v27 = &v43;
      goto LABEL_35;
    }
    v31 = &v43;
  }
  memcpy(v31, v25, v26);
  v26 = v39[0];
  v27 = v41;
LABEL_35:
  v42 = v26;
  v27->m128i_i8[v26] = 0;
  v28 = v41;
  v32[2].m128i_i8[0] &= ~1u;
  v25 = src;
  v32->m128i_i64[0] = (__int64)v32[1].m128i_i64;
  v29 = v42;
  if ( v28 == &v43 )
    goto LABEL_39;
  v32->m128i_i64[0] = (__int64)v28;
  v32[1].m128i_i64[0] = v43.m128i_i64[0];
LABEL_40:
  v32->m128i_i64[1] = v29;
  if ( v25 != dest )
    _libc_free((unsigned __int64)v25);
LABEL_29:
  if ( v47 != v49 )
    _libc_free((unsigned __int64)v47);
  return v32;
}
