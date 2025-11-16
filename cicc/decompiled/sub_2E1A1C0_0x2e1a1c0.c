// Function: sub_2E1A1C0
// Address: 0x2e1a1c0
//
__int64 __fastcall sub_2E1A1C0(__int64 a1, unsigned int *a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
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
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  unsigned int v20; // r11d
  __int64 v21; // rsi
  __int64 v22; // rcx
  _QWORD *v23; // rdx
  _QWORD *v24; // rax

  v7 = *a2;
  if ( *a2 )
  {
    v9 = v7 - 1;
    if ( *(_QWORD *)(a1 + 8 * v9 + 128) == a6 )
    {
      v16 = 16 * v9;
      v17 = (_QWORD *)(a1 + v16 + 8);
      if ( *v17 == a4 )
      {
        *a2 = v7 - 1;
        if ( v7 != a3 && *(_QWORD *)(a1 + 8LL * v7 + 128) == a6 && (v19 = (_QWORD *)(a1 + 16LL * v7), *v19 == a5) )
        {
          v20 = v7 + 1;
          for ( *(_QWORD *)(a1 + v16 + 8) = v19[1];
                a3 != v20;
                *(_QWORD *)(a1 + 8 * v22 + 128) = *(_QWORD *)(a1 + 8 * v21 + 128) )
          {
            v21 = v20;
            v22 = v20++ - 1;
            v23 = (_QWORD *)(a1 + 16 * v21);
            v24 = (_QWORD *)(a1 + 16 * v22);
            *v24 = *v23;
            v24[1] = v23[1];
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
    result = 9;
    if ( v7 == 8 )
      return result;
  }
  if ( v7 == a3 )
  {
    v18 = (_QWORD *)(a1 + 16LL * v7);
    *v18 = a4;
    v18[1] = a5;
    *(_QWORD *)(a1 + 8LL * v7 + 128) = a6;
    return v7 + 1;
  }
  else if ( *(_QWORD *)(a1 + 8LL * v7 + 128) == a6 && (v15 = (_QWORD *)(a1 + 16LL * v7), *v15 == a5) )
  {
    *v15 = a4;
    return a3;
  }
  else
  {
    result = 9;
    if ( a3 != 8 )
    {
      v11 = a3 - 1;
      do
      {
        v12 = (_QWORD *)(a1 + 16LL * v11);
        v13 = (_QWORD *)(a1 + 16LL * (v11 + 1));
        *v13 = *v12;
        v13[1] = v12[1];
        *(_QWORD *)(a1 + 8LL * (v11 + 1) + 128) = *(_QWORD *)(a1 + 8LL * v11 + 128);
        LODWORD(v13) = v11--;
      }
      while ( v7 != (_DWORD)v13 );
      v14 = (_QWORD *)(a1 + 16LL * v7);
      *v14 = a4;
      v14[1] = a5;
      result = a3 + 1;
      *(_QWORD *)(a1 + 8LL * v7 + 128) = a6;
    }
  }
  return result;
}
