// Function: sub_1E1D4C0
// Address: 0x1e1d4c0
//
__int16 __fastcall sub_1E1D4C0(__int64 a1, unsigned int a2, _QWORD *a3, char a4)
{
  __int64 v6; // rcx
  unsigned int v7; // edx
  __int16 v8; // ax
  _WORD *v9; // rdx
  __int64 v10; // rax
  _WORD *v11; // r10
  unsigned __int16 *v12; // rax
  __int16 v13; // r11
  unsigned __int16 v14; // dx
  __int64 v15; // rcx
  __int64 v16; // rax
  unsigned __int16 v17; // dx
  __int16 v18; // cx

  *(_QWORD *)(a1 + 32) = 0;
  *(_WORD *)(a1 + 24) = 0;
  *(_DWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_BYTE *)(a1 + 16) = a4;
  *(_DWORD *)(a1 + 40) = 0;
  *(_WORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  v6 = a3[6];
  v7 = *(_DWORD *)(*a3 + 24LL * a2 + 16);
  v8 = a2 * (v7 & 0xF);
  v9 = (_WORD *)(v6 + 2LL * (v7 >> 4));
  LOWORD(v10) = *v9 + v8;
  v11 = v9 + 1;
  *(_WORD *)(a1 + 24) = v10;
  *(_QWORD *)(a1 + 32) = v9 + 1;
  while ( v11 )
  {
    v12 = (unsigned __int16 *)(a3[5] + 4LL * *(unsigned __int16 *)(a1 + 24));
    v13 = *(_WORD *)(a1 + 24);
    v14 = *v12;
    for ( *(_DWORD *)(a1 + 40) = *(_DWORD *)v12; v14; *(_DWORD *)(a1 + 40) = v14 )
    {
      v15 = *(unsigned int *)(*a3 + 24LL * v14 + 8);
      v16 = a3[6];
      *(_WORD *)(a1 + 48) = v14;
      v10 = v16 + 2 * v15;
      *(_QWORD *)(a1 + 56) = v10;
      while ( v10 )
      {
        if ( a4 )
          return v10;
        v17 = *(_WORD *)(a1 + 48);
        if ( a2 != v17 )
          return v10;
        v10 += 2;
        *(_QWORD *)(a1 + 56) = v10;
        v18 = *(_WORD *)(v10 - 2);
        *(_WORD *)(a1 + 48) = v18 + v17;
        if ( !v18 )
        {
          *(_QWORD *)(a1 + 56) = 0;
          break;
        }
      }
      v14 = *(_WORD *)(a1 + 42);
    }
    *(_QWORD *)(a1 + 32) = ++v11;
    LOWORD(v10) = *(v11 - 1);
    *(_WORD *)(a1 + 24) = v10 + v13;
    if ( !(_WORD)v10 )
    {
      *(_QWORD *)(a1 + 32) = 0;
      return v10;
    }
  }
  return v10;
}
