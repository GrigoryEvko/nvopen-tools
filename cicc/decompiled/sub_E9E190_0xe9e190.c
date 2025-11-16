// Function: sub_E9E190
// Address: 0xe9e190
//
__int64 __fastcall sub_E9E190(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v10; // r12
  __int64 result; // rax
  __int64 v12; // r14
  __int64 v13; // rdx
  int v14; // ecx
  __int64 v15; // r15
  __int64 v16; // rdi
  int v17; // r15d
  unsigned __int64 v18[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = a1 + 16;
  v10 = sub_C8D7D0(a1, a1 + 16, a2, 0x30u, v18, a6);
  result = *(_QWORD *)a1;
  v12 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v12 )
  {
    v13 = v10;
    do
    {
      if ( v13 )
      {
        *(_QWORD *)v13 = *(_QWORD *)result;
        *(_QWORD *)(v13 + 8) = *(_QWORD *)(result + 8);
        *(_QWORD *)(v13 + 16) = *(_QWORD *)(result + 16);
        *(_QWORD *)(v13 + 24) = *(_QWORD *)(result + 24);
        v14 = *(_DWORD *)(result + 32);
        *(_QWORD *)(result + 24) = 0;
        *(_QWORD *)(result + 16) = 0;
        *(_QWORD *)(result + 8) = 0;
        *(_DWORD *)(v13 + 32) = v14;
        *(_QWORD *)(v13 + 40) = *(_QWORD *)(result + 40);
      }
      result += 48;
      v13 += 48;
    }
    while ( v12 != result );
    result = *(unsigned int *)(a1 + 8);
    v15 = *(_QWORD *)a1;
    v12 = *(_QWORD *)a1 + 48 * result;
    if ( *(_QWORD *)a1 != v12 )
    {
      do
      {
        v16 = *(_QWORD *)(v12 - 40);
        v12 -= 48;
        if ( v16 )
        {
          v8 = *(_QWORD *)(v12 + 24) - v16;
          result = j_j___libc_free_0(v16, v8);
        }
      }
      while ( v15 != v12 );
      v12 = *(_QWORD *)a1;
    }
  }
  v17 = v18[0];
  if ( v7 != v12 )
    result = _libc_free(v12, v8);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v17;
  return result;
}
