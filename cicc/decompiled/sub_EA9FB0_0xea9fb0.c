// Function: sub_EA9FB0
// Address: 0xea9fb0
//
__int64 __fastcall sub_EA9FB0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  __int64 v9; // rsi
  __int64 v10; // rbx
  __int64 result; // rax
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // rdi
  int v16; // r15d
  unsigned __int64 v17[7]; // [rsp+8h] [rbp-38h] BYREF

  v8 = a1 + 16;
  v9 = a1 + 16;
  v10 = sub_C8D7D0(a1, a1 + 16, a2, 0x28u, v17, a6);
  result = *(_QWORD *)a1;
  v12 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v12 )
  {
    v13 = v10;
    do
    {
      if ( v13 )
      {
        *(_DWORD *)v13 = *(_DWORD *)result;
        *(__m128i *)(v13 + 8) = _mm_loadu_si128((const __m128i *)(result + 8));
        *(_DWORD *)(v13 + 32) = *(_DWORD *)(result + 32);
        *(_QWORD *)(v13 + 24) = *(_QWORD *)(result + 24);
        *(_DWORD *)(result + 32) = 0;
      }
      result += 40;
      v13 += 40;
    }
    while ( v12 != result );
    v14 = *(_QWORD *)a1;
    result = 5LL * *(unsigned int *)(a1 + 8);
    v12 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v12 )
    {
      do
      {
        v12 -= 40;
        if ( *(_DWORD *)(v12 + 32) > 0x40u )
        {
          v15 = *(_QWORD *)(v12 + 24);
          if ( v15 )
            result = j_j___libc_free_0_0(v15);
        }
      }
      while ( v12 != v14 );
      v12 = *(_QWORD *)a1;
    }
  }
  v16 = v17[0];
  if ( v8 != v12 )
    result = _libc_free(v12, v9);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v16;
  return result;
}
