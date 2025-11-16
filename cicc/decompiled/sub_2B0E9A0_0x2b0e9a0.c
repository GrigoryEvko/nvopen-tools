// Function: sub_2B0E9A0
// Address: 0x2b0e9a0
//
__int64 __fastcall sub_2B0E9A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  void *v8; // rdi
  unsigned int v9; // r13d
  size_t v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rbx
  __int64 i; // rsi

  result = 0x600000000LL;
  v8 = a1 + 2;
  *a1 = v8;
  a1[1] = 0x600000000LL;
  v9 = *(_DWORD *)(a2 + 8);
  if ( v9 && a1 != (_QWORD *)a2 )
  {
    v10 = 8LL * v9;
    if ( v9 <= 6
      || (sub_C8D5F0((__int64)a1, v8, v9, 8u, a5, a6), v8 = (void *)*a1, (v10 = 8LL * *(unsigned int *)(a2 + 8)) != 0) )
    {
      result = (__int64)memcpy(v8, *(const void **)a2, v10);
      *((_DWORD *)a1 + 2) = v9;
      v8 = (void *)*a1;
      v11 = 8LL * v9;
      if ( v9 == 1 )
        return result;
    }
    else
    {
      *((_DWORD *)a1 + 2) = v9;
      v11 = 8LL * v9;
    }
    v12 = v11 >> 3;
    for ( i = (v12 - 2) / 2; ; --i )
    {
      result = sub_2B0E850((__int64)v8, i, v12, *((_QWORD *)v8 + i));
      if ( !i )
        break;
    }
  }
  return result;
}
