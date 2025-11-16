// Function: sub_D38730
// Address: 0xd38730
//
__int64 __fastcall sub_D38730(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  __int64 v4; // rdx
  int v5; // eax
  __int64 v6; // rdx
  __int64 result; // rax
  unsigned int v8; // r13d
  const void *v10; // rax
  const void *v11; // rsi
  size_t v12; // rdx

  v3 = (_QWORD *)(a1 + 48);
  *(v3 - 5) = 0;
  *(v3 - 4) = 0;
  *((_DWORD *)v3 - 6) = 0;
  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_DWORD *)(a2 + 24);
  *(v3 - 6) = 1;
  *(v3 - 5) = v4;
  v6 = *(_QWORD *)(a2 + 16);
  *((_DWORD *)v3 - 6) = v5;
  result = 0x100000000LL;
  *(v3 - 4) = v6;
  ++*(_QWORD *)a2;
  *(_QWORD *)(a2 + 8) = 0;
  *(_QWORD *)(a2 + 16) = 0;
  *(_DWORD *)(a2 + 24) = 0;
  *(_QWORD *)(a1 + 32) = v3;
  *(_QWORD *)(a1 + 40) = 0x100000000LL;
  v8 = *(_DWORD *)(a2 + 40);
  if ( v8 )
  {
    result = a2 + 32;
    if ( a1 + 32 != a2 + 32 )
    {
      v10 = *(const void **)(a2 + 32);
      v11 = (const void *)(a2 + 48);
      if ( v10 == v11 )
      {
        v12 = 8;
        if ( v8 == 1
          || (result = sub_C8D5F0(a1 + 32, v3, v8, 8u, a1 + 32, v8),
              v3 = *(_QWORD **)(a1 + 32),
              v11 = *(const void **)(a2 + 32),
              (v12 = 8LL * *(unsigned int *)(a2 + 40)) != 0) )
        {
          result = (__int64)memcpy(v3, v11, v12);
        }
        *(_DWORD *)(a2 + 40) = 0;
        *(_DWORD *)(a1 + 40) = v8;
      }
      else
      {
        *(_QWORD *)(a1 + 32) = v10;
        result = *(unsigned int *)(a2 + 44);
        *(_QWORD *)(a2 + 32) = v11;
        *(_QWORD *)(a2 + 40) = 0;
        *(_DWORD *)(a1 + 40) = v8;
        *(_DWORD *)(a1 + 44) = result;
      }
    }
  }
  return result;
}
