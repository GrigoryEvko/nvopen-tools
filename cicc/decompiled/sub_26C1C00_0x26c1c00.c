// Function: sub_26C1C00
// Address: 0x26c1c00
//
__int64 __fastcall sub_26C1C00(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdi
  unsigned int v8; // r13d
  void *v9; // rax
  __int64 v10; // rdx
  const void *v11; // rsi
  size_t v12; // rdx

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  result = sub_C7D6A0(0, 0, 8);
  v7 = *(unsigned int *)(a2 + 24);
  *(_DWORD *)(a1 + 24) = v7;
  if ( (_DWORD)v7 )
  {
    v9 = (void *)sub_C7D670(16 * v7, 8);
    v10 = *(unsigned int *)(a1 + 24);
    v11 = *(const void **)(a2 + 8);
    *(_QWORD *)(a1 + 8) = v9;
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    result = (__int64)memcpy(v9, v11, 16 * v10);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
  }
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  v8 = *(_DWORD *)(a2 + 40);
  if ( v8 )
  {
    result = a2 + 32;
    if ( a1 + 32 != a2 + 32 )
    {
      result = sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v8, 0x10u, v5, v6);
      v12 = 16LL * *(unsigned int *)(a2 + 40);
      if ( v12 )
        result = (__int64)memcpy(*(void **)(a1 + 32), *(const void **)(a2 + 32), v12);
      *(_DWORD *)(a1 + 40) = v8;
    }
  }
  return result;
}
