// Function: sub_80AD10
// Address: 0x80ad10
//
void __fastcall sub_80AD10(__int64 a1)
{
  char v1; // al
  __int64 v2; // rax
  __int64 v3; // rax

  if ( !unk_4D04170 )
    return;
  v1 = *(_BYTE *)(a1 + 201);
  if ( (v1 & 0x10) != 0 )
    return;
  *(_BYTE *)(a1 + 201) = v1 | 0x10;
  if ( unk_4F0697C && *(_QWORD *)(a1 + 240) )
  {
    if ( (*(_BYTE *)(a1 + 195) & 1) != 0 )
      return;
    if ( *(_BYTE *)(a1 + 172) == 2 )
      goto LABEL_13;
  }
  else
  {
    if ( *(_BYTE *)(a1 + 172) == 2 )
    {
LABEL_13:
      if ( (unsigned int)sub_736A50(a1) && ((*(_BYTE *)(a1 + 195) & 1) == 0 || qword_4F077A8 > 0xEA5Fu) )
      {
        v3 = *(_QWORD *)(a1 + 104);
        if ( !v3 || (*(_BYTE *)(v3 + 11) & 0x20) == 0 )
          goto LABEL_18;
      }
      return;
    }
    if ( (*(_BYTE *)(a1 + 195) & 1) != 0 && qword_4F077A8 <= 0xEA5Fu )
      return;
  }
  v2 = *(_QWORD *)(a1 + 104);
  if ( !v2 || (*(_BYTE *)(v2 + 11) & 0x20) == 0 )
LABEL_18:
    sub_80A450(a1, 0xBu);
}
