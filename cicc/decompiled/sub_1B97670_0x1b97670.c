// Function: sub_1B97670
// Address: 0x1b97670
//
__int64 __fastcall sub_1B97670(__int64 a1, int *a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v4; // r8
  int v5; // edi
  int v6; // esi
  unsigned int v7; // ecx
  _DWORD *v8; // rax
  int v9; // r9d
  int v10; // r11d
  _DWORD *v11; // r10

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v4 = *(_QWORD *)(a1 + 8);
    v5 = *a2;
    v6 = result - 1;
    v7 = (result - 1) & (37 * v5);
    v8 = (_DWORD *)(v4 + 16LL * v7);
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
      while ( v9 != 0x7FFFFFFF )
      {
        if ( !v11 && v9 == 0x80000000 )
          v11 = v8;
        v7 = v6 & (v10 + v7);
        v8 = (_DWORD *)(v4 + 16LL * v7);
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
