// Function: sub_2293510
// Address: 0x2293510
//
unsigned __int64 __fastcall sub_2293510(__int64 a1)
{
  unsigned __int64 result; // rax
  unsigned __int64 v3; // r8
  unsigned __int64 v4; // rdi
  int v6; // ebx
  unsigned int v7; // r11d
  unsigned int v8; // edi
  int v9; // r8d
  __int64 v10; // r12
  unsigned __int64 v11; // r10
  unsigned __int64 v12; // r10
  int v14; // esi

  result = **(_QWORD **)a1;
  v3 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  if ( (result & 1) != 0 )
  {
    v4 = result >> 58;
    result = ~(-1LL << (result >> 58)) & (-1LL << v3) & (result >> 1);
    if ( result && v4 > v3 )
    {
      __asm { tzcnt   rdx, rax }
      *(_DWORD *)(a1 + 8) = _RDX;
      return result;
    }
    goto LABEL_17;
  }
  v6 = *(_DWORD *)(result + 64);
  if ( v6 == (_DWORD)v3 || (v7 = (unsigned int)v3 >> 6, v8 = (unsigned int)(v6 - 1) >> 6, (unsigned int)v3 >> 6 > v8) )
  {
LABEL_17:
    *(_DWORD *)(a1 + 8) = -1;
    return result;
  }
  v9 = v3 & 0x3F;
  v10 = *(_QWORD *)result;
  v11 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9);
  if ( v9 == 0 )
    v11 = 0;
  result = v7;
  v12 = ~v11;
  while ( 1 )
  {
    _RDX = *(_QWORD *)(v10 + 8 * result);
    v14 = result;
    if ( v7 == (_DWORD)result )
      _RDX = v12 & *(_QWORD *)(v10 + 8 * result);
    if ( (_DWORD)result == v8 )
      break;
    if ( _RDX )
      goto LABEL_16;
    if ( v8 < (unsigned int)++result )
      goto LABEL_17;
  }
  result = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
  _RDX &= result;
  if ( !_RDX )
    goto LABEL_17;
LABEL_16:
  __asm { tzcnt   rdx, rdx }
  *(_DWORD *)(a1 + 8) = (v14 << 6) + _RDX;
  return result;
}
