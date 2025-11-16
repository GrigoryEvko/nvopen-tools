// Function: sub_27D99E0
// Address: 0x27d99e0
//
__int64 __fastcall sub_27D99E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  void *v8; // rdi
  unsigned int v9; // r13d
  const void *v11; // rax
  const void *v12; // rsi
  size_t v13; // rdx

  result = 0xA00000000LL;
  v8 = (void *)(a1 + 16);
  *(_QWORD *)a1 = v8;
  *(_QWORD *)(a1 + 8) = 0xA00000000LL;
  v9 = *(_DWORD *)(a2 + 8);
  if ( v9 && a1 != a2 )
  {
    v11 = *(const void **)a2;
    v12 = (const void *)(a2 + 16);
    if ( v11 == v12 )
    {
      v13 = 8LL * v9;
      if ( v9 <= 0xA
        || (result = sub_C8D5F0(a1, v8, v9, 8u, v9, a6),
            v8 = *(void **)a1,
            v12 = *(const void **)a2,
            (v13 = 8LL * *(unsigned int *)(a2 + 8)) != 0) )
      {
        result = (__int64)memcpy(v8, v12, v13);
      }
      *(_DWORD *)(a2 + 8) = 0;
      *(_DWORD *)(a1 + 8) = v9;
    }
    else
    {
      *(_QWORD *)a1 = v11;
      result = *(unsigned int *)(a2 + 12);
      *(_QWORD *)a2 = v12;
      *(_QWORD *)(a2 + 8) = 0;
      *(_DWORD *)(a1 + 8) = v9;
      *(_DWORD *)(a1 + 12) = result;
    }
  }
  return result;
}
