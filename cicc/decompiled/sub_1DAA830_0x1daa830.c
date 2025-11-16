// Function: sub_1DAA830
// Address: 0x1daa830
//
__int64 __fastcall sub_1DAA830(__int64 a1, unsigned int *a2, unsigned int a3, __int64 a4, __int64 a5, int a6)
{
  unsigned int v7; // r11d
  __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // r10
  _QWORD *v12; // rdx
  __int64 result; // rax
  _QWORD *v14; // rax
  unsigned int v15; // eax
  _QWORD *v16; // rcx
  _QWORD *v17; // rdx
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  unsigned int v20; // r11d
  __int64 v21; // rsi
  __int64 v22; // rcx
  _QWORD *v23; // rdx
  _QWORD *v24; // rax

  v7 = *a2;
  if ( *a2 )
  {
    v9 = v7 - 1;
    if ( ((a6 ^ *(_DWORD *)(a1 + 4 * v9 + 144)) & 0x7FFFFFFF) == 0
      && ((*(_BYTE *)(a1 + 4 * (v9 + 36) + 3) ^ HIBYTE(a6)) & 0x80u) == 0 )
    {
      v10 = 16 * v9;
      v11 = (_QWORD *)(a1 + v10 + 8);
      if ( a4 == *v11 )
      {
        *a2 = v7 - 1;
        if ( v7 == a3
          || ((*(_DWORD *)(a1 + 4LL * v7 + 144) ^ a6) & 0x7FFFFFFF) != 0
          || ((*(_BYTE *)(a1 + 4 * (v7 + 36LL) + 3) ^ HIBYTE(a6)) & 0x80u) != 0
          || (v12 = (_QWORD *)(a1 + 16LL * v7), *v12 != a5) )
        {
          *v11 = a5;
          return a3;
        }
        else
        {
          v20 = v7 + 1;
          for ( *(_QWORD *)(a1 + v10 + 8) = v12[1];
                a3 != v20;
                *(_DWORD *)(a1 + 4 * v22 + 144) = *(_DWORD *)(a1 + 4 * v21 + 144) )
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
      }
    }
    result = 10;
    if ( v7 == 9 )
      return result;
  }
  if ( v7 == a3 )
  {
    v19 = (_QWORD *)(a1 + 16LL * v7);
    *v19 = a4;
    v19[1] = a5;
    *(_DWORD *)(a1 + 4LL * v7 + 144) = a6;
    return v7 + 1;
  }
  else if ( ((a6 ^ *(_DWORD *)(a1 + 4LL * v7 + 144)) & 0x7FFFFFFF) == 0
         && ((*(_BYTE *)(a1 + 4 * (v7 + 36LL) + 3) ^ HIBYTE(a6)) & 0x80u) == 0
         && (v14 = (_QWORD *)(a1 + 16LL * v7), a5 == *v14) )
  {
    *v14 = a4;
    return a3;
  }
  else
  {
    result = 10;
    if ( a3 != 9 )
    {
      v15 = a3 - 1;
      do
      {
        v16 = (_QWORD *)(a1 + 16LL * v15);
        v17 = (_QWORD *)(a1 + 16LL * (v15 + 1));
        *v17 = *v16;
        v17[1] = v16[1];
        *(_DWORD *)(a1 + 4LL * (v15 + 1) + 144) = *(_DWORD *)(a1 + 4LL * v15 + 144);
        LODWORD(v17) = v15--;
      }
      while ( v7 != (_DWORD)v17 );
      v18 = (_QWORD *)(a1 + 16LL * v7);
      *v18 = a4;
      v18[1] = a5;
      *(_DWORD *)(a1 + 4LL * v7 + 144) = a6;
      return a3 + 1;
    }
  }
  return result;
}
