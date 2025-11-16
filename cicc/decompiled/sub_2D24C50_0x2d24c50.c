// Function: sub_2D24C50
// Address: 0x2d24c50
//
_DWORD *__fastcall sub_2D24C50(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r14
  unsigned int v8; // ebx
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // rdi
  __m128i *v13; // rsi
  __m128i *v14; // rsi
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r9
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rdi
  unsigned __int64 v21; // r8
  signed __int64 v22; // rdx
  _DWORD *v23; // r12
  __int64 v24; // r15
  _DWORD *v25; // rbx
  unsigned int v26; // eax
  _DWORD *result; // rax
  _DWORD *v28; // rdx
  _DWORD *v29; // rcx
  _DWORD *v30; // r11
  size_t v31; // r15
  _DWORD *v32; // r8
  size_t v33; // r10
  __int64 v34; // r9
  __int64 v35; // [rsp+8h] [rbp-98h]
  int v36; // [rsp+10h] [rbp-90h]
  size_t v37; // [rsp+10h] [rbp-90h]
  _DWORD *v38; // [rsp+18h] [rbp-88h]
  __m128i v39; // [rsp+20h] [rbp-80h] BYREF
  _QWORD s[6]; // [rsp+30h] [rbp-70h] BYREF
  int v41; // [rsp+60h] [rbp-40h]

  v6 = a2;
  v8 = (unsigned int)(a2 + 63) >> 6;
  v9 = v8;
  *(_DWORD *)(a1 + 80) = 0;
  *(_DWORD *)(a1 + 144) = 0;
  *(_DWORD *)(a1 + 208) = 0;
  v39.m128i_i64[0] = (__int64)s;
  v39.m128i_i64[1] = 0x600000000LL;
  if ( v8 > 6 )
  {
    sub_C8D5F0((__int64)&v39, s, v8, 8u, a5, a6);
    memset((void *)v39.m128i_i64[0], 0, 8LL * v8);
    v39.m128i_i32[2] = (unsigned int)(a2 + 63) >> 6;
  }
  else
  {
    if ( v8 )
    {
      v9 = 8LL * v8;
      if ( v9 )
        memset(s, 0, v9);
    }
    v39.m128i_i32[2] = (unsigned int)(a2 + 63) >> 6;
  }
  v41 = a2;
  sub_2D23900(a1, (char **)&v39, v9, a4, a5, a6);
  v12 = (_QWORD *)v39.m128i_i64[0];
  *(_DWORD *)(a1 + 64) = v41;
  if ( v12 != s )
    _libc_free((unsigned __int64)v12);
  v13 = *(__m128i **)(a1 + 72);
  v39 = (__m128i)1uLL;
  s[0] = 0;
  sub_2D23DB0(a1 + 72, v13, v6, &v39, v10, v11);
  v14 = *(__m128i **)(a1 + 136);
  v39 = (__m128i)1uLL;
  s[0] = 0;
  sub_2D23DB0(a1 + 136, v14, v6, &v39, v15, v16);
  v18 = *(unsigned int *)(a1 + 208);
  v19 = *(unsigned int *)(a1 + 212);
  v20 = a1 + 200;
  v21 = v6 + v18;
  v22 = 4 * v18;
  if ( 4 * v18 )
  {
    if ( v21 > v19 )
    {
      sub_C8D5F0(v20, (const void *)(a1 + 216), v6 + v18, 4u, v21, v17);
      v18 = *(unsigned int *)(a1 + 208);
      v20 = a1 + 200;
      v22 = 4 * v18;
    }
    v23 = *(_DWORD **)(a1 + 200);
    v24 = v22 >> 2;
    v25 = &v23[(unsigned __int64)v22 / 4];
    if ( v6 <= v22 >> 2 )
    {
      v30 = &v23[(unsigned __int64)v22 / 4];
      v31 = 4 * (v18 - v6);
      v32 = &v23[v31 / 4];
      v33 = v22 - v31;
      v34 = (__int64)(v22 - v31) >> 2;
      if ( v18 + v34 > (unsigned __int64)*(unsigned int *)(a1 + 212) )
      {
        v35 = (__int64)(v22 - v31) >> 2;
        v37 = v22 - v31;
        sub_C8D5F0(v20, (const void *)(a1 + 216), v18 + v34, 4u, (__int64)v32, v34);
        v18 = *(unsigned int *)(a1 + 208);
        LODWORD(v34) = v35;
        v33 = v37;
        v32 = &v23[v31 / 4];
        v30 = (_DWORD *)(*(_QWORD *)(a1 + 200) + 4 * v18);
      }
      if ( v25 != v32 )
      {
        v36 = v34;
        v38 = v32;
        memmove(v30, v32, v33);
        LODWORD(v18) = *(_DWORD *)(a1 + 208);
        LODWORD(v34) = v36;
        v32 = v38;
      }
      *(_DWORD *)(a1 + 208) = v34 + v18;
      if ( v23 != v32 )
        memmove(&v25[v31 / 0xFFFFFFFFFFFFFFFCLL], v23, v31);
      result = &v23[v6];
      if ( v6 )
      {
        do
          *v23++ = 2;
        while ( result != v23 );
      }
    }
    else
    {
      v26 = v6 + v18;
      *(_DWORD *)(a1 + 208) = v26;
      if ( v23 != v25 )
      {
        memcpy(&v23[v26 + v22 / 0xFFFFFFFFFFFFFFFCLL], v23, v22);
        if ( v24 )
        {
          do
            *v23++ = 2;
          while ( v25 != v23 );
        }
      }
      for ( result = &v25[v6 - v24]; result != v25; ++v25 )
        *v25 = 2;
    }
  }
  else
  {
    if ( v21 > v19 )
    {
      sub_C8D5F0(v20, (const void *)(a1 + 216), v6 + v18, 4u, v21, v17);
      v18 = *(unsigned int *)(a1 + 208);
      v22 = 4 * v18;
    }
    if ( v6 )
    {
      v28 = (_DWORD *)(*(_QWORD *)(a1 + 200) + v22);
      v29 = &v28[v6];
      if ( v28 != v29 )
      {
        do
          *v28++ = 2;
        while ( v29 != v28 );
        v18 = *(unsigned int *)(a1 + 208);
      }
    }
    result = (_DWORD *)(v6 + v18);
    *(_DWORD *)(a1 + 208) = (_DWORD)result;
  }
  return result;
}
