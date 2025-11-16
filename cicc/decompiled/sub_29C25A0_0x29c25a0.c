// Function: sub_29C25A0
// Address: 0x29c25a0
//
void __fastcall sub_29C25A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rax
  int v12; // edx
  __int64 *v13; // rax
  __m128i *v14; // r15
  __m128i *v15; // rax
  unsigned __int64 v16; // rdi
  int v17; // eax
  __m128i *v18; // r15
  __m128i *v19; // rax
  unsigned __int64 v20; // rdi
  int v21; // r12d
  int v22; // [rsp+8h] [rbp-48h]
  unsigned __int64 v23[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = *(unsigned int *)(a2 + 296);
  if ( *(_DWORD *)(a2 + 300) <= (unsigned int)v8 )
  {
    v14 = (__m128i *)sub_C8D7D0(a2 + 288, a2 + 304, 0, 0x20u, v23, a6);
    v15 = &v14[2 * *(unsigned int *)(a2 + 296)];
    if ( v15 )
    {
      v15->m128i_i64[0] = a1;
      v15->m128i_i64[1] = a3;
      v15[1].m128i_i64[1] = (__int64)off_4CDFC50 + 2;
    }
    sub_BC3E80(a2 + 288, v14);
    v16 = *(_QWORD *)(a2 + 288);
    v17 = v23[0];
    if ( a2 + 304 != v16 )
    {
      v22 = v23[0];
      _libc_free(v16);
      v17 = v22;
    }
    ++*(_DWORD *)(a2 + 296);
    *(_QWORD *)(a2 + 288) = v14;
    *(_DWORD *)(a2 + 300) = v17;
  }
  else
  {
    v9 = *(_DWORD *)(a2 + 296);
    v10 = (__int64 *)(*(_QWORD *)(a2 + 288) + 32 * v8);
    if ( v10 )
    {
      *v10 = a1;
      v10[1] = a3;
      v10[3] = (__int64)off_4CDFC50 + 2;
      v9 = *(_DWORD *)(a2 + 296);
    }
    *(_DWORD *)(a2 + 296) = v9 + 1;
  }
  v11 = *(unsigned int *)(a2 + 440);
  v12 = v11;
  if ( *(_DWORD *)(a2 + 444) <= (unsigned int)v11 )
  {
    v18 = (__m128i *)sub_C8D7D0(a2 + 432, a2 + 448, 0, 0x20u, v23, a6);
    v19 = &v18[2 * *(unsigned int *)(a2 + 440)];
    if ( v19 )
    {
      v19->m128i_i64[0] = a1;
      v19->m128i_i64[1] = a3;
      v19[1].m128i_i64[1] = (__int64)&off_4CDFC58 + 2;
    }
    sub_BC3FC0(a2 + 432, v18);
    v20 = *(_QWORD *)(a2 + 432);
    v21 = v23[0];
    if ( a2 + 448 != v20 )
      _libc_free(v20);
    *(_QWORD *)(a2 + 432) = v18;
    *(_DWORD *)(a2 + 444) = v21;
    ++*(_DWORD *)(a2 + 440);
  }
  else
  {
    v13 = (__int64 *)(*(_QWORD *)(a2 + 432) + 32 * v11);
    if ( v13 )
    {
      *v13 = a1;
      v13[1] = a3;
      v13[3] = (__int64)&off_4CDFC58 + 2;
      v12 = *(_DWORD *)(a2 + 440);
    }
    *(_DWORD *)(a2 + 440) = v12 + 1;
  }
}
