// Function: sub_ADDA00
// Address: 0xadda00
//
__int64 __fastcall sub_ADDA00(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  unsigned int v7; // r13d
  __int64 v9; // rdi
  __int64 v10; // rdx
  size_t v11; // rdx
  int v12; // edx

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v3 = *(_QWORD *)(a2 + 8);
  result = *(unsigned int *)(a2 + 24);
  ++*(_QWORD *)a2;
  *(_QWORD *)(a1 + 8) = v3;
  v5 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a2 + 8) = 0;
  *(_QWORD *)(a2 + 16) = 0;
  *(_DWORD *)(a2 + 24) = 0;
  v6 = a1 + 48;
  *(_QWORD *)a1 = 1;
  *(_QWORD *)(a1 + 16) = v5;
  *(_DWORD *)(a1 + 24) = result;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0;
  v7 = *(_DWORD *)(a2 + 40);
  if ( v7 )
  {
    result = a2 + 32;
    v9 = a1 + 32;
    if ( v9 != a2 + 32 )
    {
      v10 = *(_QWORD *)(a2 + 32);
      result = a2 + 48;
      if ( v10 == a2 + 48 )
      {
        result = sub_C8D5F0(v9, v6, v7, 8);
        v11 = 8LL * *(unsigned int *)(a2 + 40);
        if ( v11 )
          result = (__int64)memcpy(*(void **)(a1 + 32), *(const void **)(a2 + 32), v11);
        *(_DWORD *)(a1 + 40) = v7;
        *(_DWORD *)(a2 + 40) = 0;
      }
      else
      {
        *(_QWORD *)(a1 + 32) = v10;
        v12 = *(_DWORD *)(a2 + 44);
        *(_DWORD *)(a1 + 40) = v7;
        *(_DWORD *)(a1 + 44) = v12;
        *(_QWORD *)(a2 + 32) = result;
        *(_QWORD *)(a2 + 40) = 0;
      }
    }
  }
  return result;
}
