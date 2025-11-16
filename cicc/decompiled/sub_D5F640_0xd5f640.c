// Function: sub_D5F640
// Address: 0xd5f640
//
__int64 __fastcall sub_D5F640(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  __int64 v8; // rsi
  __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rcx
  __int64 v14; // r12
  __int64 v15; // rcx
  int v16; // esi
  __int64 v17; // r15
  __int64 v18; // rdi
  int v19; // r15d
  unsigned __int64 v20[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = a1 + 16;
  result = sub_C8D7D0(a1, a1 + 16, a2, 0x20u, v20, a6);
  v11 = *(_QWORD *)a1;
  v12 = result;
  v13 = 32LL * *(unsigned int *)(a1 + 8);
  v14 = *(_QWORD *)a1 + v13;
  if ( *(_QWORD *)a1 != v14 )
  {
    v15 = result + v13;
    do
    {
      if ( result )
      {
        *(_DWORD *)(result + 8) = *(_DWORD *)(v11 + 8);
        *(_QWORD *)result = *(_QWORD *)v11;
        v16 = *(_DWORD *)(v11 + 24);
        *(_DWORD *)(v11 + 8) = 0;
        *(_DWORD *)(result + 24) = v16;
        v8 = *(_QWORD *)(v11 + 16);
        *(_QWORD *)(result + 16) = v8;
        *(_DWORD *)(v11 + 24) = 0;
      }
      result += 32;
      v11 += 32;
    }
    while ( result != v15 );
    v17 = *(_QWORD *)a1;
    v14 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v14 )
    {
      do
      {
        v14 -= 32;
        if ( *(_DWORD *)(v14 + 24) > 0x40u )
        {
          v18 = *(_QWORD *)(v14 + 16);
          if ( v18 )
            result = j_j___libc_free_0_0(v18);
        }
        if ( *(_DWORD *)(v14 + 8) > 0x40u && *(_QWORD *)v14 )
          result = j_j___libc_free_0_0(*(_QWORD *)v14);
      }
      while ( v14 != v17 );
      v14 = *(_QWORD *)a1;
    }
  }
  v19 = v20[0];
  if ( v7 != v14 )
    result = _libc_free(v14, v8);
  *(_QWORD *)a1 = v12;
  *(_DWORD *)(a1 + 12) = v19;
  return result;
}
