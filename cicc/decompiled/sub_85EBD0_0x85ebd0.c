// Function: sub_85EBD0
// Address: 0x85ebd0
//
__int64 __fastcall sub_85EBD0(__int64 a1, _DWORD *a2)
{
  int v2; // ecx
  unsigned int v3; // r8d
  char v4; // al
  __int64 v5; // rdx
  __int64 i; // rax
  __int64 v8; // rax

  v2 = *(_DWORD *)(a1 + 40);
  v3 = 0;
  if ( v2 == unk_4F066A8 )
    return v3;
  if ( v2 == -1 )
    return (unsigned int)-1;
  v4 = *(_BYTE *)(a1 + 80);
  if ( (unsigned __int8)(v4 - 4) <= 1u )
  {
    v8 = *(_QWORD *)(a1 + 88);
    if ( v8 && (*(_BYTE *)(v8 + 177) & 0x30) == 0x10 )
      return (unsigned int)-1;
  }
  else
  {
    if ( ((v4 - 7) & 0xFD) != 0 )
    {
      if ( v4 != 3 )
        goto LABEL_7;
      v5 = *(_QWORD *)(a1 + 88);
      if ( !v5 )
        goto LABEL_7;
    }
    else
    {
      v5 = *(_QWORD *)(a1 + 88);
      if ( !v5 || (*(_BYTE *)(v5 + 170) & 0x10) == 0 )
        goto LABEL_7;
      if ( **(_QWORD **)(v5 + 216) )
        return (unsigned int)-1;
      if ( v4 != 3 )
        goto LABEL_7;
    }
    if ( *(_BYTE *)(v5 + 140) == 12 && *(_BYTE *)(v5 + 184) == 10 )
      return (unsigned int)-1;
  }
LABEL_7:
  v3 = dword_4F04C5C;
  if ( v2 == *(_DWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C) )
  {
    if ( dword_4F04C58 != -1 || dword_4F04C38 )
    {
      *a2 = 1;
      return (unsigned int)dword_4F04C5C;
    }
    return v3;
  }
  v3 = dword_4F04C64;
  if ( dword_4F04C64 < 0 )
    return (unsigned int)-1;
  for ( i = 776LL * dword_4F04C64 + qword_4F04C68[0]; (*(_BYTE *)(i + 4) & 0xFD) == 5 || v2 != *(_DWORD *)i; i -= 776 )
  {
    if ( --v3 == -1 )
      return v3;
  }
  if ( *(_DWORD *)(i + 400) == -1 && (*(_BYTE *)(i + 5) & 8) == 0 )
    return v3;
  *a2 = 1;
  return v3;
}
