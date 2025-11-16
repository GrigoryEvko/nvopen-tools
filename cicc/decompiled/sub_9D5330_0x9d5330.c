// Function: sub_9D5330
// Address: 0x9d5330
//
__int64 __fastcall sub_9D5330(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rsi
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rcx
  __int64 v10; // r12
  __int64 v11; // rcx
  int v12; // esi
  __int64 v13; // r15
  __int64 v14; // rdi
  int v15; // r15d
  __int64 v16[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = a1 + 16;
  v4 = a1 + 16;
  result = sub_C8D7D0(a1, a1 + 16, a2, 32, v16);
  v7 = *(_QWORD *)a1;
  v8 = result;
  v9 = 32LL * *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1 + v9;
  if ( *(_QWORD *)a1 != v10 )
  {
    v11 = result + v9;
    do
    {
      if ( result )
      {
        *(_DWORD *)(result + 8) = *(_DWORD *)(v7 + 8);
        *(_QWORD *)result = *(_QWORD *)v7;
        v12 = *(_DWORD *)(v7 + 24);
        *(_DWORD *)(v7 + 8) = 0;
        *(_DWORD *)(result + 24) = v12;
        v4 = *(_QWORD *)(v7 + 16);
        *(_QWORD *)(result + 16) = v4;
        *(_DWORD *)(v7 + 24) = 0;
      }
      result += 32;
      v7 += 32;
    }
    while ( result != v11 );
    v13 = *(_QWORD *)a1;
    v10 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v10 -= 32;
        if ( *(_DWORD *)(v10 + 24) > 0x40u )
        {
          v14 = *(_QWORD *)(v10 + 16);
          if ( v14 )
            result = j_j___libc_free_0_0(v14);
        }
        if ( *(_DWORD *)(v10 + 8) > 0x40u && *(_QWORD *)v10 )
          result = j_j___libc_free_0_0(*(_QWORD *)v10);
      }
      while ( v10 != v13 );
      v10 = *(_QWORD *)a1;
    }
  }
  v15 = v16[0];
  if ( v3 != v10 )
    result = _libc_free(v10, v4);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v15;
  return result;
}
