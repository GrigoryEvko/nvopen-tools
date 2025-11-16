// Function: sub_1D49910
// Address: 0x1d49910
//
__int64 __fastcall sub_1D49910(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6)
{
  char v8; // r11
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdi

  if ( !a5 )
    return 0;
  v8 = a6;
  v9 = (unsigned int)(*(_DWORD *)(a4 + 60) - 1);
  if ( *(_BYTE *)(*(_QWORD *)(a4 + 40) + 16 * v9) == 111 )
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(a4 + 48);
      if ( !v10 )
        break;
      while ( (_DWORD)v9 != *(_DWORD *)(v10 + 8) )
      {
        v10 = *(_QWORD *)(v10 + 32);
        if ( !v10 )
          return (unsigned int)sub_1D47A00(a4, a1, a3, v8) ^ 1;
      }
      v11 = *(_QWORD *)(v10 + 16);
      if ( !v11 )
        break;
      v8 = 0;
      v9 = (unsigned int)(*(_DWORD *)(v11 + 60) - 1);
      if ( *(_BYTE *)(*(_QWORD *)(v11 + 40) + 16 * v9) != 111 )
        return (unsigned int)sub_1D47A00(v11, a1, a3, 0) ^ 1;
      a4 = *(_QWORD *)(v10 + 16);
    }
  }
  else
  {
    v8 = a6;
  }
  return (unsigned int)sub_1D47A00(a4, a1, a3, v8) ^ 1;
}
