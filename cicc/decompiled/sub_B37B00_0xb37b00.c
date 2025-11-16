// Function: sub_B37B00
// Address: 0xb37b00
//
__int64 __fastcall sub_B37B00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v6; // rbx
  __m128i v7; // xmm0
  _QWORD *v8; // rax
  _QWORD *v9; // rdi
  unsigned __int64 v10; // rdx
  char *v11; // rax
  char *v12; // r8
  unsigned int v13; // ebx
  unsigned int v14; // ebx
  unsigned int v15; // eax
  __int64 v16; // rcx
  __int64 v17; // rcx
  void *v18; // rax
  void *v19; // rsi
  unsigned __int64 v20; // rdx
  __int64 v21; // rax
  signed __int64 v22; // rdx
  char *v23; // rcx
  size_t v24; // r8
  char *v25; // rax
  __int64 *v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // r13
  signed __int64 v30; // [rsp+8h] [rbp-E8h]
  size_t v31; // [rsp+8h] [rbp-E8h]
  _QWORD v32[4]; // [rsp+20h] [rbp-D0h] BYREF
  __m128i *v33; // [rsp+40h] [rbp-B0h]
  __int64 v34; // [rsp+48h] [rbp-A8h]
  __m128i v35; // [rsp+50h] [rbp-A0h] BYREF
  void *src; // [rsp+60h] [rbp-90h]
  _BYTE *v37; // [rsp+68h] [rbp-88h]
  char *v38; // [rsp+70h] [rbp-80h]
  __int64 v39[2]; // [rsp+80h] [rbp-70h] BYREF
  __m128i v40; // [rsp+90h] [rbp-60h] BYREF
  char *v41; // [rsp+A0h] [rbp-50h]
  char *v42; // [rsp+A8h] [rbp-48h]
  char *v43; // [rsp+B0h] [rbp-40h]

  v6 = 2;
  v32[0] = a3;
  v32[1] = a4;
  if ( a5 )
  {
    v32[2] = a5;
    v6 = 3;
  }
  strcpy(v40.m128i_i8, "align");
  v7 = _mm_load_si128(&v40);
  v39[0] = (__int64)&v40;
  v33 = &v35;
  v34 = 5;
  v39[1] = 0;
  v40.m128i_i8[0] = 0;
  src = 0;
  v37 = 0;
  v38 = 0;
  v35 = v7;
  v8 = (_QWORD *)sub_22077B0(v6 * 8);
  v9 = &v8[v6];
  src = v8;
  v38 = (char *)&v8[v6];
  *v8 = v32[0];
  v8[v6 - 1] = v32[v6 - 1];
  v10 = (unsigned __int64)(v8 + 1) & 0xFFFFFFFFFFFFFFF8LL;
  v11 = (char *)v8 - v10;
  v12 = (char *)((char *)v32 - v11);
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
  v37 = v9;
  if ( (__m128i *)v39[0] != &v40 )
    j_j___libc_free_0(v39[0], v40.m128i_i64[0] + 1);
  v39[0] = (__int64)&v40;
  sub_B32E10(v39, v33, (__int64)v33->m128i_i64 + v34);
  v18 = v37;
  v19 = src;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v20 = v37 - (_BYTE *)src;
  if ( v37 == src )
  {
    v24 = 0;
    v22 = 0;
    v23 = 0;
  }
  else
  {
    if ( v20 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(v39, src, v20, v17);
    v30 = v37 - (_BYTE *)src;
    v21 = sub_22077B0(v37 - (_BYTE *)src);
    v19 = src;
    v22 = v30;
    v23 = (char *)v21;
    v18 = v37;
    v24 = v37 - (_BYTE *)src;
  }
  v41 = v23;
  v42 = v23;
  v43 = &v23[v22];
  if ( v19 != v18 )
  {
    v31 = v24;
    v25 = (char *)memmove(v23, v19, v24);
    v24 = v31;
    v23 = v25;
  }
  v26 = *(__int64 **)(a1 + 72);
  v42 = &v23[v24];
  v27 = sub_ACD6D0(v26);
  v28 = sub_B33B40(a1, v27, (__int64)v39, 1);
  if ( v41 )
    j_j___libc_free_0(v41, v43 - v41);
  if ( (__m128i *)v39[0] != &v40 )
    j_j___libc_free_0(v39[0], v40.m128i_i64[0] + 1);
  if ( src )
    j_j___libc_free_0(src, v38 - (_BYTE *)src);
  if ( v33 != &v35 )
    j_j___libc_free_0(v33, v35.m128i_i64[0] + 1);
  return v28;
}
