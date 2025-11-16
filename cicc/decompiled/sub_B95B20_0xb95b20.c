// Function: sub_B95B20
// Address: 0xb95b20
//
__int64 __fastcall sub_B95B20(__int64 a1, __int64 *a2, __int64 **a3)
{
  __int64 result; // rax
  __int64 v4; // rcx
  int v5; // eax
  __int64 v6; // r9
  int v7; // r11d
  __int64 *v8; // r10
  unsigned int v9; // r8d
  __int64 *v10; // rdi
  __int64 v11; // rsi
  __int64 *v12; // rbx

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v4 = *a2;
    v5 = result - 1;
    v6 = *(_QWORD *)(a1 + 8);
    v7 = 1;
    v8 = 0;
    v9 = v5 & *(_DWORD *)(*a2 + 4);
    v10 = (__int64 *)(v6 + 8LL * v9);
    v11 = *v10;
    if ( v4 == *v10 )
    {
      *a3 = v10;
      return 1;
    }
    else
    {
      while ( v11 != -4096 )
      {
        if ( v11 != -8192 || v8 )
          v10 = v8;
        v9 = v5 & (v7 + v9);
        v12 = (__int64 *)(v6 + 8LL * v9);
        v11 = *v12;
        if ( *v12 == v4 )
        {
          *a3 = v12;
          return 1;
        }
        ++v7;
        v8 = v10;
        v10 = (__int64 *)(v6 + 8LL * v9);
      }
      if ( !v8 )
        v8 = v10;
      *a3 = v8;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
  }
  return result;
}
