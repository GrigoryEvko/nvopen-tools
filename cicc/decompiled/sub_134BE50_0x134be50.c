// Function: sub_134BE50
// Address: 0x134be50
//
_QWORD *__fastcall sub_134BE50(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned int v5; // eax
  char v6; // cl
  unsigned int v7; // eax
  unsigned __int64 v8; // rax
  __int64 v9; // rsi
  char v10; // cl
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned __int64 v13; // rdx
  int v14; // eax
  unsigned int v16; // edx

  v3 = sub_130FA30(a2);
  if ( v3 > 0x7000000000000000LL )
  {
    v9 = a1 + 1024;
    v11 = 4;
    if ( *(_QWORD *)(a1 + 1048) & 0xFFFFFFFFFFFFFF80LL )
    {
      __asm { tzcnt   rdx, rdx }
      v16 = _RDX + 192;
      return sub_1348AB0((_QWORD *)(a1 + 16LL * v16));
    }
    goto LABEL_10;
  }
  _BitScanReverse64(&v4, v3);
  v5 = v4 - ((((v3 - 1) & v3) == 0) - 1);
  if ( v5 < 0xE )
    v5 = 14;
  v6 = v5 - 3;
  v7 = v5 - 14;
  if ( !v7 )
    v6 = 12;
  v8 = (((v3 - 1) >> v6) & 3) + 4 * v7;
  v9 = a1 + 1024;
  v10 = v8;
  v11 = v8 >> 6;
  v12 = v11;
  v13 = *(_QWORD *)(a1 + 8 * v11 + 1024) & -(1LL << v10);
  if ( v13 )
  {
LABEL_11:
    v14 = (_DWORD)v11 << 6;
    if ( !_BitScanForward64(&v13, v13) )
      LODWORD(v13) = -1;
    v16 = v13 + v14;
    if ( v16 == 64 )
      return *(_QWORD **)(a1 + 4224);
    return sub_1348AB0((_QWORD *)(a1 + 16LL * v16));
  }
  ++v11;
  if ( v12 )
  {
LABEL_10:
    while ( 1 )
    {
      v13 = *(_QWORD *)(v9 + 8 * v11);
      if ( v13 )
        break;
      ++v11;
    }
    goto LABEL_11;
  }
  return *(_QWORD **)(a1 + 4224);
}
