// Function: sub_14AF300
// Address: 0x14af300
//
__int64 __fastcall sub_14AF300(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v4; // rcx
  int v5; // eax
  _QWORD *v6; // rsi
  unsigned __int64 v8; // rdx
  void *v9; // rcx
  __int64 v10; // rax

  v2 = *(unsigned __int8 *)(a2 + 16);
  v3 = 0;
  if ( (unsigned __int8)v2 <= 0x17u )
  {
    if ( (_BYTE)v2 != 5 )
      return v3;
    v8 = *(unsigned __int16 *)(a2 + 18);
    if ( (unsigned __int16)v8 > 0x17u )
      return v3;
    v9 = &loc_80A800;
    v5 = (unsigned __int16)v8;
    if ( !_bittest64((const __int64 *)&v9, v8) )
      return v3;
  }
  else
  {
    if ( (unsigned __int8)v2 > 0x2Fu )
      return v3;
    v4 = 0x80A800000000LL;
    if ( !_bittest64(&v4, v2) )
      return v3;
    v5 = (unsigned __int8)v2 - 24;
  }
  v3 = 0;
  if ( v5 == 11 && (*(_BYTE *)(a2 + 17) & 4) != 0 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v6 = *(_QWORD **)(a2 - 8);
    else
      v6 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v3 = 0;
    if ( *v6 == *(_QWORD *)a1 )
    {
      v10 = v6[3];
      if ( *(_BYTE *)(v10 + 16) == 13 )
      {
        v3 = 1;
        **(_QWORD **)(a1 + 8) = v10;
      }
    }
  }
  return v3;
}
