// Function: sub_1523EE0
// Address: 0x1523ee0
//
__int64 __fastcall sub_1523EE0(__int64 a1)
{
  unsigned __int64 v1; // rdx
  __int64 v2; // rax
  __int64 result; // rax
  char v4; // al
  unsigned __int8 v5; // dl
  char v6; // dl
  unsigned __int64 v7; // rdx
  void *v8; // rcx
  char v9; // al

  v1 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned __int8)v1 > 0x17u )
  {
    if ( (unsigned __int8)v1 <= 0x2Fu )
    {
      v2 = 0x80A800000000LL;
      if ( _bittest64(&v2, v1) )
        return (*(_BYTE *)(a1 + 17) >> 1) & 3;
    }
    if ( (unsigned int)(unsigned __int8)v1 - 41 > 1 && (unsigned __int8)(v1 - 48) > 1u )
    {
      v4 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
      if ( v4 == 16 )
        v4 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 8LL);
      if ( (unsigned __int8)(v4 - 1) > 5u && (_BYTE)v1 != 76 )
        return 0;
      goto LABEL_11;
    }
    return (*(_BYTE *)(a1 + 17) & 2) != 0;
  }
  result = 0;
  if ( (_BYTE)v1 != 5 )
    return result;
  v7 = *(unsigned __int16 *)(a1 + 18);
  if ( (unsigned __int16)v7 <= 0x17u )
  {
    v8 = &loc_80A800;
    if ( _bittest64((const __int64 *)&v8, v7) )
      return (*(_BYTE *)(a1 + 17) >> 1) & 3;
  }
  if ( (unsigned int)(unsigned __int16)v7 - 17 <= 1 || (unsigned __int16)(v7 - 24) <= 1u )
    return (*(_BYTE *)(a1 + 17) & 2) != 0;
  v9 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
  if ( v9 == 16 )
    v9 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 8LL);
  if ( (unsigned __int8)(v9 - 1) <= 5u || (result = 0, (_WORD)v7 == 52) )
  {
LABEL_11:
    v5 = *(_BYTE *)(a1 + 17);
    result = (unsigned __int8)(v5 << 6) & 0x80;
    v6 = v5 >> 1;
    if ( (v6 & 2) != 0 )
      result |= 2uLL;
    if ( (v6 & 4) != 0 )
      result |= 4uLL;
    if ( (v6 & 8) != 0 )
      result |= 8uLL;
    if ( (v6 & 0x10) != 0 )
      result |= 0x10uLL;
    if ( (v6 & 0x20) != 0 )
      result |= 0x20uLL;
    if ( (v6 & 0x40) != 0 )
      return result | 0x40;
  }
  return result;
}
