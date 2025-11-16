// Function: sub_34A32D0
// Address: 0x34a32d0
//
__int64 __fastcall sub_34A32D0(__int64 a1, unsigned int *a2, unsigned int a3, __int64 a4, __int64 a5, char a6)
{
  unsigned int v7; // r11d
  __int64 v9; // rax
  __int64 result; // rax
  unsigned int v11; // eax
  _QWORD *v12; // rcx
  _QWORD *v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // rdx
  unsigned int v18; // r11d
  __int64 v19; // rsi
  __int64 v20; // rcx
  _QWORD *v21; // rdx
  _QWORD *v22; // rax
  _QWORD *v23; // rax

  v7 = *a2;
  if ( *a2 )
  {
    v9 = v7 - 1;
    if ( *(_BYTE *)(a1 + v9 + 176) == a6 )
    {
      v16 = a1 + 16 * v9;
      if ( *(_QWORD *)(v16 + 8) + 1LL == a4 )
      {
        *a2 = v7 - 1;
        if ( v7 != a3 && *(_BYTE *)(a1 + v7 + 176) == a6 && (v17 = (_QWORD *)(a1 + 16LL * v7), *v17 == a5 + 1) )
        {
          v18 = v7 + 1;
          for ( *(_QWORD *)(v16 + 8) = v17[1]; a3 != v18; *(_BYTE *)(a1 + v20 + 176) = *(_BYTE *)(a1 + v19 + 176) )
          {
            v19 = v18;
            v20 = v18++ - 1;
            v21 = (_QWORD *)(a1 + 16 * v19);
            v22 = (_QWORD *)(a1 + 16 * v20);
            *v22 = *v21;
            v22[1] = v21[1];
          }
          return a3 - 1;
        }
        else
        {
          *(_QWORD *)(v16 + 8) = a5;
          return a3;
        }
      }
    }
    result = 12;
    if ( v7 == 11 )
      return result;
  }
  if ( v7 == a3 )
  {
    v23 = (_QWORD *)(a1 + 16LL * v7);
    *v23 = a4;
    v23[1] = a5;
    *(_BYTE *)(a1 + v7 + 176) = a6;
    return v7 + 1;
  }
  else if ( *(_BYTE *)(a1 + v7 + 176) == a6 && (v15 = (_QWORD *)(a1 + 16LL * v7), a5 + 1 == *v15) )
  {
    *v15 = a4;
    return a3;
  }
  else
  {
    result = 12;
    if ( a3 != 11 )
    {
      v11 = a3 - 1;
      do
      {
        v12 = (_QWORD *)(a1 + 16LL * v11);
        v13 = (_QWORD *)(a1 + 16LL * (v11 + 1));
        *v13 = *v12;
        v13[1] = v12[1];
        *(_BYTE *)(a1 + v11 + 1 + 176) = *(_BYTE *)(a1 + v11 + 176);
        LODWORD(v13) = v11--;
      }
      while ( v7 != (_DWORD)v13 );
      v14 = (_QWORD *)(a1 + 16LL * v7);
      *v14 = a4;
      v14[1] = a5;
      result = a3 + 1;
      *(_BYTE *)(a1 + v7 + 176) = a6;
    }
  }
  return result;
}
