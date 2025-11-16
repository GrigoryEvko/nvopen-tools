// Function: sub_32365A0
// Address: 0x32365a0
//
__int64 __fastcall sub_32365A0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  unsigned int v10; // r15d
  __m128i **v12; // rdi
  unsigned __int64 v13; // rsi
  const __m128i *v14; // rcx
  __m128i *v15; // r12
  unsigned __int64 v16; // r9
  const __m128i *v17; // r13
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  char *v20; // rdi
  size_t v21; // rdx
  char *v22; // rax
  __int64 v23; // r13
  __int64 i; // r15
  __int8 v25; // dl
  bool v26; // zf
  __int64 v27; // rax
  __m128i *v28; // rbx
  __m128i *v29; // r12
  unsigned __int64 v30; // rdi
  void *v31; // rax
  const void *v32; // rsi
  const __m128i *v33; // [rsp+8h] [rbp-B8h]
  const __m128i *v34; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v35; // [rsp+10h] [rbp-B0h]
  size_t v36; // [rsp+10h] [rbp-B0h]
  __m128i *v37; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v38; // [rsp+27h] [rbp-99h] BYREF
  int v39; // [rsp+28h] [rbp-98h] BYREF
  int v40; // [rsp+2Ch] [rbp-94h] BYREF
  __m128i v41; // [rsp+30h] [rbp-90h] BYREF
  __m128i v42; // [rsp+40h] [rbp-80h] BYREF
  __m128i v43; // [rsp+50h] [rbp-70h] BYREF
  __int64 v44; // [rsp+60h] [rbp-60h]
  __int64 v45; // [rsp+68h] [rbp-58h]
  __int64 v46; // [rsp+70h] [rbp-50h]
  __int64 v47; // [rsp+78h] [rbp-48h]
  __m128i *v48; // [rsp+80h] [rbp-40h] BYREF
  __int64 v49; // [rsp+88h] [rbp-38h]
  _BYTE v50[48]; // [rsp+90h] [rbp-30h] BYREF

  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  sub_C7D6A0(0, 0, 8);
  v9 = *(unsigned int *)(a2 + 120);
  LODWORD(v47) = v9;
  if ( (_DWORD)v9 )
  {
    v31 = (void *)sub_C7D670(24 * v9, 8);
    v32 = *(const void **)(a2 + 104);
    v45 = (__int64)v31;
    v46 = *(_QWORD *)(a2 + 112);
    memcpy(v31, v32, 24LL * (unsigned int)v47);
  }
  else
  {
    v45 = 0;
    v46 = 0;
  }
  v10 = *(_DWORD *)(a2 + 136);
  v49 = 0;
  v48 = (__m128i *)v50;
  if ( v10 )
  {
    v12 = &v48;
    v13 = v10;
    sub_3226A20((__int64)&v48, v10, v5, v6, v7, v8);
    v14 = *(const __m128i **)(a2 + 128);
    v15 = v48;
    v16 = (unsigned __int64)*(unsigned int *)(a2 + 136) << 6;
    v17 = (const __m128i *)((char *)v14 + v16);
    if ( v14 != (const __m128i *)&v14->m128i_i8[v16] )
    {
      do
      {
        if ( v15 )
        {
          *v15 = _mm_loadu_si128(v14);
          v15[1].m128i_i64[0] = v14[1].m128i_i64[0];
          v15[1].m128i_i32[2] = v14[1].m128i_i32[2];
          v18 = v14[2].m128i_i64[1] - v14[2].m128i_i64[0];
          v15[2].m128i_i64[0] = 0;
          v15[2].m128i_i64[1] = 0;
          v15[3].m128i_i64[0] = 0;
          if ( v18 )
          {
            if ( v18 > 0x7FFFFFFFFFFFFFF8LL )
              sub_4261EA(v12, v13, v18);
            v33 = v14;
            v35 = v18;
            v19 = sub_22077B0(v18);
            v18 = v35;
            v14 = v33;
            v20 = (char *)v19;
          }
          else
          {
            v20 = 0;
          }
          v15[2].m128i_i64[0] = (__int64)v20;
          v15[3].m128i_i64[0] = (__int64)&v20[v18];
          v15[2].m128i_i64[1] = (__int64)v20;
          v13 = v14[2].m128i_u64[0];
          v21 = v14[2].m128i_i64[1] - v13;
          if ( v14[2].m128i_i64[1] != v13 )
          {
            v34 = v14;
            v36 = v14[2].m128i_i64[1] - v13;
            v22 = (char *)memmove(v20, (const void *)v13, v21);
            v14 = v34;
            v21 = v36;
            v20 = v22;
          }
          v12 = (__m128i **)&v20[v21];
          v15[2].m128i_i64[1] = (__int64)v12;
          v15[3].m128i_i64[1] = v14[3].m128i_i64[1];
        }
        v14 += 4;
        v15 += 4;
      }
      while ( v17 != v14 );
      v15 = v48;
    }
    LODWORD(v49) = v10;
    v37 = &v15[4 * (unsigned __int64)v10];
    do
    {
      v23 = v15[2].m128i_i64[1];
      for ( i = v15[2].m128i_i64[0]; v23 != i; i += 8 )
      {
        v27 = *(_QWORD *)i;
        v26 = *(_BYTE *)(*(_QWORD *)i + 32LL) == 0;
        v38 = *(_BYTE *)(*(_QWORD *)i + 43LL) >> 7;
        v40 = *(_DWORD *)(v27 + 44);
        v39 = *(unsigned __int16 *)(v27 + 40);
        if ( v26 )
        {
          v25 = 0;
          v42 = (__m128i)(unsigned __int64)v2;
          v43 = _mm_loadu_si128(&v42);
        }
        else
        {
          v2 = *(_QWORD *)(v27 + 24);
          v25 = 1;
        }
        v43.m128i_i64[0] = v2;
        v43.m128i_i8[8] = v25;
        v42 = _mm_loadu_si128(&v43);
        v26 = *(_BYTE *)(v27 + 16) == 1;
        v41 = v42;
        if ( !v26 )
          abort();
        v42.m128i_i64[0] = *(_QWORD *)(v27 + 8);
        sub_32363E0(a1, v15[1].m128i_i64[0], v42.m128i_i64, &v41, &v39, &v40, &v38);
      }
      v15 += 4;
    }
    while ( v15 != v37 );
    v28 = v48;
    v29 = &v48[4 * (unsigned __int64)(unsigned int)v49];
    if ( v48 != v29 )
    {
      do
      {
        v30 = v29[-2].m128i_u64[0];
        v29 -= 4;
        if ( v30 )
          j_j___libc_free_0(v30);
      }
      while ( v28 != v29 );
      v29 = v48;
    }
    if ( v29 != (__m128i *)v50 )
      _libc_free((unsigned __int64)v29);
  }
  return sub_C7D6A0(v45, 24LL * (unsigned int)v47, 8);
}
