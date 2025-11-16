// Function: sub_25BFBE0
// Address: 0x25bfbe0
//
__int64 __fastcall sub_25BFBE0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 result; // rax
  void *v6; // rdi
  unsigned int v7; // r13d
  void *v8; // rax
  __int64 v9; // rdx
  const void *v10; // rsi
  size_t v11; // rdx

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  sub_C7D6A0(0, 0, 8);
  v4 = *(unsigned int *)(a2 + 24);
  *(_DWORD *)(a1 + 24) = v4;
  if ( (_DWORD)v4 )
  {
    v8 = (void *)sub_C7D670(8 * v4, 8);
    v9 = *(unsigned int *)(a1 + 24);
    v10 = *(const void **)(a2 + 8);
    *(_QWORD *)(a1 + 8) = v8;
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    memcpy(v8, v10, 8 * v9);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
  }
  result = 0x800000000LL;
  v6 = (void *)(a1 + 48);
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0x800000000LL;
  v7 = *(_DWORD *)(a2 + 40);
  if ( v7 )
  {
    result = a2 + 32;
    if ( a1 + 32 != a2 + 32 )
    {
      v11 = 8LL * v7;
      if ( v7 <= 8
        || (result = sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v7, 8u, a1 + 32, v7),
            v6 = *(void **)(a1 + 32),
            (v11 = 8LL * *(unsigned int *)(a2 + 40)) != 0) )
      {
        result = (__int64)memcpy(v6, *(const void **)(a2 + 32), v11);
      }
      *(_DWORD *)(a1 + 40) = v7;
    }
  }
  return result;
}
