// Function: sub_26EC0A0
// Address: 0x26ec0a0
//
void __fastcall sub_26EC0A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // edx
  __int64 *v8; // rax
  __m128i *v9; // r14
  __m128i *v10; // rax
  unsigned __int64 v11; // rdi
  int v12; // r12d
  unsigned __int64 v13; // [rsp-40h] [rbp-40h] BYREF

  if ( byte_4FF8BE8 )
  {
    v6 = *(unsigned int *)(a2 + 440);
    v7 = v6;
    if ( *(_DWORD *)(a2 + 444) <= (unsigned int)v6 )
    {
      v9 = (__m128i *)sub_C8D7D0(a2 + 432, a2 + 448, 0, 0x20u, &v13, a6);
      v10 = &v9[2 * *(unsigned int *)(a2 + 440)];
      if ( v10 )
      {
        v10->m128i_i64[0] = a1;
        v10[1].m128i_i64[1] = (__int64)off_4CDFC48 + 2;
      }
      sub_BC3FC0(a2 + 432, v9);
      v11 = *(_QWORD *)(a2 + 432);
      v12 = v13;
      if ( a2 + 448 != v11 )
        _libc_free(v11);
      ++*(_DWORD *)(a2 + 440);
      *(_QWORD *)(a2 + 432) = v9;
      *(_DWORD *)(a2 + 444) = v12;
    }
    else
    {
      v8 = (__int64 *)(*(_QWORD *)(a2 + 432) + 32 * v6);
      if ( v8 )
      {
        *v8 = a1;
        v8[3] = (__int64)off_4CDFC48 + 2;
        v7 = *(_DWORD *)(a2 + 440);
      }
      *(_DWORD *)(a2 + 440) = v7 + 1;
    }
  }
}
