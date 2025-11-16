// Function: sub_1781CC0
// Address: 0x1781cc0
//
bool __fastcall sub_1781CC0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rdx
  bool result; // al
  __int64 v4; // r8
  int v5; // edx
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  unsigned __int64 v9; // rcx
  void *v10; // r8

  v2 = *(unsigned __int8 *)(a2 + 16);
  result = 0;
  if ( (unsigned __int8)v2 <= 0x17u )
  {
    if ( (_BYTE)v2 != 5 )
      return result;
    v9 = *(unsigned __int16 *)(a2 + 18);
    if ( (unsigned __int16)v9 > 0x17u )
      return result;
    v10 = &loc_80A800;
    v5 = (unsigned __int16)v9;
    if ( !_bittest64((const __int64 *)&v10, v9) )
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
  if ( v5 == 15 )
  {
    result = (*(_BYTE *)(a2 + 17) & 2) != 0;
    if ( (*(_BYTE *)(a2 + 17) & 2) != 0 )
    {
      v6 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
         ? *(__int64 **)(a2 - 8)
         : (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v7 = *v6;
      result = 0;
      if ( v7 )
      {
        **(_QWORD **)a1 = v7;
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v8 = *(_QWORD *)(a2 - 8);
        else
          v8 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        return *(_QWORD *)(v8 + 24) == *(_QWORD *)(a1 + 8);
      }
    }
  }
  return result;
}
