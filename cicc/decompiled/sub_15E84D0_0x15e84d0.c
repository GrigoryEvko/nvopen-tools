// Function: sub_15E84D0
// Address: 0x15e84d0
//
_QWORD *__fastcall sub_15E84D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v6; // rbx
  __m128i v7; // xmm0
  _QWORD *v8; // rax
  __m128i *v9; // rdi
  unsigned __int64 v10; // rdx
  char *v11; // rax
  char *v12; // r8
  unsigned int v13; // ebx
  unsigned int v14; // ebx
  unsigned int v15; // eax
  __int64 v16; // rcx
  __m128i *v17; // r8
  size_t v18; // rbx
  __m128i *v19; // rax
  __m128i *v20; // rax
  __m128i *v21; // rsi
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  char *v24; // rcx
  size_t v25; // rdx
  char *v26; // rax
  __int64 *v27; // rdi
  __int64 v28; // rax
  _QWORD *v29; // r14
  __int64 v31; // rax
  void *v32; // [rsp+8h] [rbp-F8h]
  __m128i *v33; // [rsp+8h] [rbp-F8h]
  size_t v34; // [rsp+18h] [rbp-E8h] BYREF
  _QWORD *v35; // [rsp+20h] [rbp-E0h]
  __int64 v36; // [rsp+28h] [rbp-D8h]
  _QWORD v37[4]; // [rsp+30h] [rbp-D0h] BYREF
  __m128i *v38; // [rsp+50h] [rbp-B0h]
  size_t n; // [rsp+58h] [rbp-A8h]
  __m128i v40; // [rsp+60h] [rbp-A0h] BYREF
  void *src; // [rsp+70h] [rbp-90h]
  __m128i *v42; // [rsp+78h] [rbp-88h]
  char *v43; // [rsp+80h] [rbp-80h]
  __m128i *v44; // [rsp+90h] [rbp-70h] BYREF
  size_t v45; // [rsp+98h] [rbp-68h]
  __m128i v46; // [rsp+A0h] [rbp-60h] BYREF
  char *v47; // [rsp+B0h] [rbp-50h]
  char *v48; // [rsp+B8h] [rbp-48h]
  char *v49; // [rsp+C0h] [rbp-40h]

  v6 = 2;
  v35 = v37;
  v37[0] = a3;
  v37[1] = a4;
  v36 = 0x400000002LL;
  if ( a5 )
  {
    v37[2] = a5;
    v6 = 3;
    LODWORD(v36) = 3;
  }
  strcpy(v46.m128i_i8, "align");
  v7 = _mm_load_si128(&v46);
  v44 = &v46;
  v38 = &v40;
  n = 5;
  v45 = 0;
  v46.m128i_i8[0] = 0;
  src = 0;
  v42 = 0;
  v43 = 0;
  v40 = v7;
  v8 = (_QWORD *)sub_22077B0(v6 * 8);
  v9 = (__m128i *)&v8[v6];
  src = v8;
  v43 = (char *)&v8[v6];
  *v8 = v37[0];
  v8[v6 - 1] = v37[v6 - 1];
  v10 = (unsigned __int64)(v8 + 1) & 0xFFFFFFFFFFFFFFF8LL;
  v11 = (char *)v8 - v10;
  v12 = (char *)((char *)v37 - v11);
  v13 = ((_DWORD)v11 + v6 * 8) & 0xFFFFFFF8;
  if ( v13 >= 8 )
  {
    v14 = v13 & 0xFFFFFFF8;
    v15 = 0;
    do
    {
      v16 = v15;
      v15 += 8;
      *(_QWORD *)(v10 + v16) = *(_QWORD *)&v12[v16];
    }
    while ( v15 < v14 );
  }
  v42 = v9;
  if ( v44 != &v46 )
  {
    v9 = v44;
    j_j___libc_free_0(v44, v46.m128i_i64[0] + 1);
  }
  v17 = v38;
  v18 = n;
  v44 = &v46;
  if ( &v38->m128i_i8[n] && !v38 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v34 = n;
  if ( n > 0xF )
  {
    v33 = v38;
    v31 = sub_22409D0(&v44, &v34, 0);
    v17 = v33;
    v44 = (__m128i *)v31;
    v9 = (__m128i *)v31;
    v46.m128i_i64[0] = v34;
  }
  else
  {
    if ( n == 1 )
    {
      v46.m128i_i8[0] = v38->m128i_i8[0];
      v19 = &v46;
      goto LABEL_13;
    }
    if ( !n )
    {
      v19 = &v46;
      goto LABEL_13;
    }
    v9 = &v46;
  }
  memcpy(v9, v17, v18);
  v18 = v34;
  v19 = v44;
LABEL_13:
  v45 = v18;
  v19->m128i_i8[v18] = 0;
  v20 = v42;
  v21 = (__m128i *)src;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v22 = (char *)v42 - (_BYTE *)src;
  if ( v42 == src )
  {
    v25 = 0;
    v22 = 0;
    v24 = 0;
  }
  else
  {
    if ( v22 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(v9, src, v10);
    v23 = sub_22077B0((char *)v42 - (_BYTE *)src);
    v21 = (__m128i *)src;
    v24 = (char *)v23;
    v20 = v42;
    v25 = (char *)v42 - (_BYTE *)src;
  }
  v47 = v24;
  v48 = v24;
  v49 = &v24[v22];
  if ( v21 != v20 )
  {
    v32 = (void *)v25;
    v26 = (char *)memmove(v24, v21, v25);
    v25 = (size_t)v32;
    v24 = v26;
  }
  v27 = (__int64 *)a1[3];
  v48 = &v24[v25];
  v28 = sub_159C4F0(v27);
  v29 = sub_15E7F40(a1, v28, (__int64)&v44, 1);
  if ( v47 )
    j_j___libc_free_0(v47, v49 - v47);
  if ( v44 != &v46 )
    j_j___libc_free_0(v44, v46.m128i_i64[0] + 1);
  if ( src )
    j_j___libc_free_0(src, v43 - (_BYTE *)src);
  if ( v38 != &v40 )
    j_j___libc_free_0(v38, v40.m128i_i64[0] + 1);
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
  return v29;
}
