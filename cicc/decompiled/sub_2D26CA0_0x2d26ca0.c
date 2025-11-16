// Function: sub_2D26CA0
// Address: 0x2d26ca0
//
__int64 __fastcall sub_2D26CA0(__int64 *a1, unsigned int a2, int a3, char a4)
{
  unsigned int v5; // r11d
  unsigned int v6; // edi
  char v7; // bl
  int v8; // esi
  __int64 v9; // r12
  __int64 v10; // rdx
  unsigned __int64 v11; // r9
  unsigned __int64 v12; // r9

  if ( a2 == a3 )
    return 0xFFFFFFFFLL;
  v5 = a2 >> 6;
  v6 = (unsigned int)(a3 - 1) >> 6;
  v7 = a3;
  if ( a2 >> 6 > v6 )
    return 0xFFFFFFFFLL;
  v8 = a2 & 0x3F;
  v9 = *a1;
  v10 = v5;
  v11 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v8);
  if ( v8 == 0 )
    v11 = 0;
  v12 = ~v11;
  while ( 1 )
  {
    _RAX = *(_QWORD *)(v9 + 8 * v10);
    if ( !a4 )
      _RAX = ~*(_QWORD *)(v9 + 8 * v10);
    if ( v5 == (_DWORD)v10 )
      _RAX &= v12;
    if ( v6 == (_DWORD)v10 )
      break;
    if ( _RAX )
      goto LABEL_14;
    if ( v6 < (unsigned int)++v10 )
      return 0xFFFFFFFFLL;
  }
  _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -v7;
  if ( !_RAX )
    return 0xFFFFFFFFLL;
LABEL_14:
  __asm { tzcnt   rax, rax }
  return (unsigned int)(((_DWORD)v10 << 6) + _RAX);
}
