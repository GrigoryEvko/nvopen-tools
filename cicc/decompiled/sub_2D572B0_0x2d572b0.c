// Function: sub_2D572B0
// Address: 0x2d572b0
//
__int64 __fastcall sub_2D572B0(__int64 a1)
{
  __int64 v1; // rsi
  _BYTE *v3; // r11
  __int64 v4; // rax
  int v5; // edx
  int v6; // r9d
  _BYTE *v7; // rcx
  _BYTE *v8; // r8
  _BYTE *v9; // rcx
  __int64 v10; // rbx

  v1 = *(_QWORD *)(a1 - 64);
  if ( *(_BYTE *)(*(_QWORD *)(v1 + 8) + 8LL) != 12 )
    return 0;
  if ( *(_BYTE *)v1 <= 0x15u )
    return 0;
  v3 = *(_BYTE **)(a1 - 32);
  if ( *v3 <= 0x15u )
    return 0;
  if ( v3 == (_BYTE *)v1 )
    return 0;
  v4 = *(_QWORD *)(v1 + 16);
  if ( !v4 )
    return 0;
  v5 = 128;
  v6 = 0;
  v7 = *(_BYTE **)(v4 + 24);
  while ( 1 )
  {
    if ( *v7 == 44 )
    {
      v8 = (_BYTE *)*((_QWORD *)v7 - 8);
      if ( v3 == v8 && v8 && (v10 = *((_QWORD *)v7 - 4), v1 == v10) && v10 )
      {
        ++v6;
      }
      else if ( (_BYTE *)v1 == v8 && v8 )
      {
        v9 = (_BYTE *)*((_QWORD *)v7 - 4);
        if ( v9 )
          v6 = (v3 != v9) + v6 - 1;
      }
    }
    v4 = *(_QWORD *)(v4 + 8);
    if ( !v4 )
      break;
    v7 = *(_BYTE **)(v4 + 24);
    if ( !--v5 )
      return 0;
  }
  if ( v6 <= 0 )
    return 0;
  sub_B53070(a1);
  return 1;
}
