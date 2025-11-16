// Function: sub_217D950
// Address: 0x217d950
//
__int64 __fastcall sub_217D950(__int64 *a1, unsigned int a2, int a3)
{
  char v4; // r11
  unsigned int v5; // r10d
  unsigned int v6; // edi
  int v7; // esi
  unsigned __int64 v8; // r9
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // r9

  if ( a2 == a3 )
    return 0xFFFFFFFFLL;
  v4 = a3;
  v5 = a2 >> 6;
  v6 = (unsigned int)(a3 - 1) >> 6;
  if ( a2 >> 6 > v6 )
    return 0xFFFFFFFFLL;
  v7 = a2 & 0x3F;
  v8 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v7);
  v9 = *a1;
  if ( v7 == 0 )
    v8 = 0;
  v10 = v5;
  v11 = ~v8;
  while ( 1 )
  {
    _RDX = *(_QWORD *)(v9 + 8 * v10);
    if ( v5 == (_DWORD)v10 )
      _RDX = v11 & *(_QWORD *)(v9 + 8 * v10);
    if ( (_DWORD)v10 == v6 )
      break;
    if ( _RDX )
      goto LABEL_12;
    if ( v6 < (unsigned int)++v10 )
      return 0xFFFFFFFFLL;
  }
  _RDX &= 0xFFFFFFFFFFFFFFFFLL >> -v4;
  if ( _RDX )
  {
LABEL_12:
    __asm { tzcnt   rdx, rdx }
    return (unsigned int)(((_DWORD)v10 << 6) + _RDX);
  }
  return 0xFFFFFFFFLL;
}
