// Function: sub_2900F20
// Address: 0x2900f20
//
__int64 __fastcall sub_2900F20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 result; // rax
  __int64 v9; // rdx
  const void *v10; // rsi
  unsigned int v11; // r13d
  __int64 v13; // rdi
  __int64 v14; // rdx
  size_t v15; // rdx
  int v16; // edx

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v7 = *(_QWORD *)(a2 + 8);
  result = *(unsigned int *)(a2 + 24);
  ++*(_QWORD *)a2;
  *(_QWORD *)(a1 + 8) = v7;
  v9 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a2 + 8) = 0;
  *(_QWORD *)(a2 + 16) = 0;
  *(_DWORD *)(a2 + 24) = 0;
  v10 = (const void *)(a1 + 48);
  *(_QWORD *)a1 = 1;
  *(_QWORD *)(a1 + 16) = v9;
  *(_DWORD *)(a1 + 24) = result;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0;
  v11 = *(_DWORD *)(a2 + 40);
  if ( v11 )
  {
    result = a2 + 32;
    v13 = a1 + 32;
    if ( v13 != a2 + 32 )
    {
      v14 = *(_QWORD *)(a2 + 32);
      result = a2 + 48;
      if ( v14 == a2 + 48 )
      {
        result = sub_C8D5F0(v13, v10, v11, 8u, a5, a6);
        v15 = 8LL * *(unsigned int *)(a2 + 40);
        if ( v15 )
          result = (__int64)memcpy(*(void **)(a1 + 32), *(const void **)(a2 + 32), v15);
        *(_DWORD *)(a1 + 40) = v11;
        *(_DWORD *)(a2 + 40) = 0;
      }
      else
      {
        *(_QWORD *)(a1 + 32) = v14;
        v16 = *(_DWORD *)(a2 + 44);
        *(_DWORD *)(a1 + 40) = v11;
        *(_DWORD *)(a1 + 44) = v16;
        *(_QWORD *)(a2 + 32) = result;
        *(_QWORD *)(a2 + 40) = 0;
      }
    }
  }
  return result;
}
