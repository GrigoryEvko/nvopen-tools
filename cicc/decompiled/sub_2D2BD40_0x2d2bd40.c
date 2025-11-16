// Function: sub_2D2BD40
// Address: 0x2d2bd40
//
__int64 __fastcall sub_2D2BD40(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v4; // r8
  __int64 v5; // rdi
  int v6; // esi
  unsigned int v7; // ecx
  _QWORD *v8; // rax
  __int64 v9; // r9
  int v10; // r11d
  _QWORD *v11; // r10

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v4 = *(_QWORD *)(a1 + 8);
    v5 = *a2;
    v6 = result - 1;
    v7 = (result - 1) & (37 * v5);
    v8 = (_QWORD *)(v4 + 16LL * v7);
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
      while ( v9 != -4096 )
      {
        if ( !v11 && v9 == -8192 )
          v11 = v8;
        v7 = v6 & (v10 + v7);
        v8 = (_QWORD *)(v4 + 16LL * v7);
        v9 = *v8;
        if ( v5 == *v8 )
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
