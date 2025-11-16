// Function: sub_8EE9E0
// Address: 0x8ee9e0
//
_BOOL8 __fastcall sub_8EE9E0(__int64 a1, int a2, int a3)
{
  int v3; // r9d
  int v6; // r9d
  int v7; // eax
  int v8; // edx
  _BYTE *v9; // rcx
  _BYTE *v10; // rsi
  int v11; // edx
  bool v12; // sf
  int v13; // eax
  _BYTE *v14; // r8
  int v15; // esi
  int v16; // eax
  int v18; // edx
  int v19; // eax

  v3 = a2 + 14;
  if ( a2 + 7 >= 0 )
    v3 = a2 + 7;
  v6 = v3 >> 3;
  v7 = 1 << (a3 % 8);
  v8 = a3 / 8;
  if ( v8 < v6 - 1 )
  {
    if ( !v7 )
      return 0;
    v9 = (_BYTE *)(a1 + v8);
    v10 = &v9[v6 - v8 - 2];
    while ( 1 )
    {
      v11 = (unsigned __int8)*v9;
      v12 = v11 + v7 < 0;
      v13 = v11 + v7;
      *v9 = v13;
      if ( v12 )
        v13 += 255;
      v7 = v13 >> 8;
      if ( v9 == v10 )
        break;
      ++v9;
      if ( !v7 )
        return 0;
    }
  }
  if ( !v7 )
    return 0;
  v14 = (_BYTE *)(a1 + v6 - 1);
  v15 = (unsigned __int8)*v14;
  if ( (a2 & 7) != 0 )
  {
    v18 = ~(-1 << (a2 % 8));
    v19 = (unsigned __int8)(v18 & v15) + v7;
    *v14 = v19 & ~(-1 << (a2 % 8));
    return (v19 & ~(unsigned __int8)v18) != 0;
  }
  else
  {
    v16 = v15 + v7;
    *v14 = v16;
    return (v16 & 0xFFFFFF00) != 0;
  }
}
