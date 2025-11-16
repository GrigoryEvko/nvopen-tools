// Function: sub_264AC00
// Address: 0x264ac00
//
__int64 __fastcall sub_264AC00(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v4; // r9
  __int64 v5; // rdi
  int v6; // esi
  unsigned __int64 v7; // rdi
  unsigned int v8; // r8d
  _QWORD *v9; // rax
  unsigned __int64 v10; // rcx
  int v11; // r11d
  _QWORD *v12; // r10

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v4 = *(_QWORD *)(a1 + 8);
    v5 = *a2;
    v6 = result - 1;
    v7 = v5 & 0xFFFFFFFFFFFFFFF8LL;
    v8 = v7 & (result - 1);
    v9 = (_QWORD *)(v4 + 144LL * v8);
    v10 = *v9 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v7 == v10 )
    {
      *a3 = v9;
      return 1;
    }
    else
    {
      v11 = 1;
      v12 = 0;
      while ( v10 != -8 )
      {
        if ( !v12 && v10 == -16 )
          v12 = v9;
        v8 = v6 & (v11 + v8);
        v9 = (_QWORD *)(v4 + 144LL * v8);
        v10 = *v9 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v7 == v10 )
        {
          *a3 = v9;
          return 1;
        }
        ++v11;
      }
      if ( !v12 )
        v12 = v9;
      *a3 = v12;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
  }
  return result;
}
