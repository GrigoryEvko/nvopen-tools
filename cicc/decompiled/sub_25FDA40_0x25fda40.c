// Function: sub_25FDA40
// Address: 0x25fda40
//
__int64 __fastcall sub_25FDA40(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 result; // rax
  int v4; // ecx
  __int64 v5; // rsi
  __int64 v6; // r9
  unsigned int v7; // r8d
  _QWORD *v8; // rax
  __int64 v9; // rdi
  int v10; // r11d
  _QWORD *v11; // r10

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v4 = result - 1;
    v5 = *a2;
    v6 = *(_QWORD *)(a1 + 8);
    v7 = v5 & (result - 1);
    v8 = (_QWORD *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( v5 == *v8 )
    {
      *a3 = v8;
      return 1;
    }
    else
    {
      v10 = 1;
      v11 = 0;
      while ( v9 != -1 )
      {
        if ( !v11 && v9 == -2 )
          v11 = v8;
        v7 = v4 & (v10 + v7);
        v8 = (_QWORD *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( *v8 == v5 )
        {
          *a3 = v8;
          return 1;
        }
        ++v10;
      }
      if ( !v11 )
        v11 = v8;
      *a3 = v11;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
  }
  return result;
}
