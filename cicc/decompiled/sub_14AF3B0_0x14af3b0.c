// Function: sub_14AF3B0
// Address: 0x14af3b0
//
_BOOL8 __fastcall sub_14AF3B0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rdx
  _BOOL8 result; // rax
  __int64 v4; // r8
  int v5; // edx
  _QWORD *v6; // rsi
  unsigned __int64 v7; // rcx
  void *v8; // r8
  __int64 v9; // rdx

  v2 = *(unsigned __int8 *)(a2 + 16);
  result = 0;
  if ( (unsigned __int8)v2 <= 0x17u )
  {
    if ( (_BYTE)v2 != 5 )
      return result;
    v7 = *(unsigned __int16 *)(a2 + 18);
    if ( (unsigned __int16)v7 > 0x17u )
      return result;
    v8 = &loc_80A800;
    v5 = (unsigned __int16)v7;
    if ( !_bittest64((const __int64 *)&v8, v7) )
      return result;
  }
  else
  {
    if ( (unsigned __int8)v2 > 0x2Fu )
      return result;
    v4 = 0x80A800000000LL;
    if ( !_bittest64(&v4, v2) )
      return result;
    v5 = (unsigned __int8)v2 - 24;
  }
  result = 0;
  if ( v5 == 11 )
  {
    result = (*(_BYTE *)(a2 + 17) & 2) != 0;
    if ( (*(_BYTE *)(a2 + 17) & 2) != 0 )
    {
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v6 = *(_QWORD **)(a2 - 8);
      else
        v6 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      if ( *v6 == *(_QWORD *)a1 && (v9 = v6[3], *(_BYTE *)(v9 + 16) == 13) )
        **(_QWORD **)(a1 + 8) = v9;
      else
        return 0;
    }
  }
  return result;
}
