// Function: sub_1F15E30
// Address: 0x1f15e30
//
__int64 __fastcall sub_1F15E30(__int64 a1, unsigned int *a2, unsigned int a3, __int64 a4, __int64 a5, int a6)
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
  _QWORD *v17; // rcx
  _QWORD *v18; // rdx
  unsigned int v19; // r11d
  __int64 v20; // rsi
  __int64 v21; // rcx
  _QWORD *v22; // rdx
  _QWORD *v23; // rax
  _QWORD *v24; // rax

  v7 = *a2;
  if ( *a2 )
  {
    v9 = v7 - 1;
    if ( *(_DWORD *)(a1 + 4 * v9 + 144) == a6 )
    {
      v16 = 16 * v9;
      v17 = (_QWORD *)(a1 + v16 + 8);
      if ( *v17 == a4 )
      {
        *a2 = v7 - 1;
        if ( v7 != a3 && *(_DWORD *)(a1 + 4LL * v7 + 144) == a6 && (v18 = (_QWORD *)(a1 + 16LL * v7), *v18 == a5) )
        {
          v19 = v7 + 1;
          for ( *(_QWORD *)(a1 + v16 + 8) = v18[1];
                a3 != v19;
                *(_DWORD *)(a1 + 4 * v21 + 144) = *(_DWORD *)(a1 + 4 * v20 + 144) )
          {
            v20 = v19;
            v21 = v19++ - 1;
            v22 = (_QWORD *)(a1 + 16 * v20);
            v23 = (_QWORD *)(a1 + 16 * v21);
            *v23 = *v22;
            v23[1] = v22[1];
          }
          return a3 - 1;
        }
        else
        {
          *v17 = a5;
          return a3;
        }
      }
    }
    result = 10;
    if ( v7 == 9 )
      return result;
  }
  if ( v7 == a3 )
  {
    v24 = (_QWORD *)(a1 + 16LL * v7);
    *v24 = a4;
    v24[1] = a5;
    *(_DWORD *)(a1 + 4LL * v7 + 144) = a6;
    return v7 + 1;
  }
  else if ( *(_DWORD *)(a1 + 4LL * v7 + 144) == a6 && (v15 = (_QWORD *)(a1 + 16LL * v7), *v15 == a5) )
  {
    *v15 = a4;
    return a3;
  }
  else
  {
    result = 10;
    if ( a3 != 9 )
    {
      v11 = a3 - 1;
      do
      {
        v12 = (_QWORD *)(a1 + 16LL * v11);
        v13 = (_QWORD *)(a1 + 16LL * (v11 + 1));
        *v13 = *v12;
        v13[1] = v12[1];
        *(_DWORD *)(a1 + 4LL * (v11 + 1) + 144) = *(_DWORD *)(a1 + 4LL * v11 + 144);
        LODWORD(v13) = v11--;
      }
      while ( v7 != (_DWORD)v13 );
      v14 = (_QWORD *)(a1 + 16LL * v7);
      *v14 = a4;
      v14[1] = a5;
      result = a3 + 1;
      *(_DWORD *)(a1 + 4LL * v7 + 144) = a6;
    }
  }
  return result;
}
