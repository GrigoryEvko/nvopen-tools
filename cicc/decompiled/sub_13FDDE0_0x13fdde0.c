// Function: sub_13FDDE0
// Address: 0x13fdde0
//
__int64 __fastcall sub_13FDDE0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v4; // r8
  int v5; // esi
  __int64 v6; // rdi
  unsigned int v7; // ecx
  _QWORD *v8; // rax
  __int64 v9; // r9
  int v10; // r11d
  _QWORD *v11; // r10

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v4 = *a2;
    v5 = result - 1;
    v6 = *(_QWORD *)(a1 + 8);
    v7 = (result - 1) & (((unsigned int)v4 >> 4) ^ ((unsigned int)v4 >> 9));
    v8 = (_QWORD *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( v4 == *v8 )
    {
      *a3 = v8;
      return 1;
    }
    else
    {
      v10 = 1;
      v11 = 0;
      while ( v9 != -8 )
      {
        if ( v9 == -16 && !v11 )
          v11 = v8;
        v7 = v5 & (v10 + v7);
        v8 = (_QWORD *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( *v8 == v4 )
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
