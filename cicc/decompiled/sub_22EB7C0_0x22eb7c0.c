// Function: sub_22EB7C0
// Address: 0x22eb7c0
//
void __fastcall sub_22EB7C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // rax
  int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // rax
  int v14; // edx
  __int64 *v15; // rax
  __m128i *v16; // r14
  __m128i *v17; // rax
  unsigned __int64 v18; // rdi
  int v19; // r15d
  __m128i *v20; // r14
  __m128i *v21; // rax
  unsigned __int64 v22; // rdi
  int v23; // r15d
  __m128i *v24; // r14
  __m128i *v25; // rax
  unsigned __int64 v26; // rdi
  int v27; // r12d
  unsigned __int64 v28[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( *(_BYTE *)(a1 + 8) )
  {
    v7 = *(unsigned int *)(a2 + 296);
    v8 = v7;
    if ( *(_DWORD *)(a2 + 300) <= (unsigned int)v7 )
    {
      v16 = (__m128i *)sub_C8D7D0(a2 + 288, a2 + 304, 0, 0x20u, v28, a6);
      v17 = &v16[2 * *(unsigned int *)(a2 + 296)];
      if ( v17 )
      {
        v17->m128i_i64[0] = a1;
        v17[1].m128i_i64[1] = (__int64)off_4CDFBD8 + 2;
      }
      sub_BC3E80(a2 + 288, v16);
      v18 = *(_QWORD *)(a2 + 288);
      v19 = v28[0];
      if ( a2 + 304 != v18 )
        _libc_free(v18);
      ++*(_DWORD *)(a2 + 296);
      *(_QWORD *)(a2 + 288) = v16;
      *(_DWORD *)(a2 + 300) = v19;
    }
    else
    {
      v9 = (__int64 *)(*(_QWORD *)(a2 + 288) + 32 * v7);
      if ( v9 )
      {
        *v9 = a1;
        v9[3] = (__int64)off_4CDFBD8 + 2;
        v8 = *(_DWORD *)(a2 + 296);
      }
      *(_DWORD *)(a2 + 296) = v8 + 1;
    }
    v10 = *(unsigned int *)(a2 + 440);
    v11 = v10;
    if ( *(_DWORD *)(a2 + 444) <= (unsigned int)v10 )
    {
      v20 = (__m128i *)sub_C8D7D0(a2 + 432, a2 + 448, 0, 0x20u, v28, a6);
      v21 = &v20[2 * *(unsigned int *)(a2 + 440)];
      if ( v21 )
      {
        v21->m128i_i64[0] = a1;
        v21[1].m128i_i64[1] = (__int64)off_4CDFBE8 + 2;
      }
      sub_BC3FC0(a2 + 432, v20);
      v22 = *(_QWORD *)(a2 + 432);
      v23 = v28[0];
      if ( a2 + 448 != v22 )
        _libc_free(v22);
      ++*(_DWORD *)(a2 + 440);
      *(_QWORD *)(a2 + 432) = v20;
      *(_DWORD *)(a2 + 444) = v23;
    }
    else
    {
      v12 = (__int64 *)(*(_QWORD *)(a2 + 432) + 32 * v10);
      if ( v12 )
      {
        *v12 = a1;
        v12[3] = (__int64)off_4CDFBE8 + 2;
        v11 = *(_DWORD *)(a2 + 440);
      }
      *(_DWORD *)(a2 + 440) = v11 + 1;
    }
    v13 = *(unsigned int *)(a2 + 584);
    v14 = v13;
    if ( *(_DWORD *)(a2 + 588) <= (unsigned int)v13 )
    {
      v24 = (__m128i *)sub_C8D7D0(a2 + 576, a2 + 592, 0, 0x20u, v28, a6);
      v25 = &v24[2 * *(unsigned int *)(a2 + 584)];
      if ( v25 )
      {
        v25->m128i_i64[0] = a1;
        v25[1].m128i_i64[1] = (__int64)off_4CDFBE0 + 2;
      }
      sub_BC4100(a2 + 576, v24);
      v26 = *(_QWORD *)(a2 + 576);
      v27 = v28[0];
      if ( a2 + 592 != v26 )
        _libc_free(v26);
      ++*(_DWORD *)(a2 + 584);
      *(_QWORD *)(a2 + 576) = v24;
      *(_DWORD *)(a2 + 588) = v27;
    }
    else
    {
      v15 = (__int64 *)(*(_QWORD *)(a2 + 576) + 32 * v13);
      if ( v15 )
      {
        *v15 = a1;
        v15[3] = (__int64)off_4CDFBE0 + 2;
        v14 = *(_DWORD *)(a2 + 584);
      }
      *(_DWORD *)(a2 + 584) = v14 + 1;
    }
  }
}
