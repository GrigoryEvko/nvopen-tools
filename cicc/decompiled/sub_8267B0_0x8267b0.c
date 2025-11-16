// Function: sub_8267B0
// Address: 0x8267b0
//
void __fastcall sub_8267B0(__int64 a1, _DWORD *a2, _DWORD *a3, _BOOL4 *a4)
{
  _BOOL4 v4; // eax

  if ( *(_BYTE *)(a1 + 172) == 2
    || (*a2 = (*(_BYTE *)(a1 + 196) & 0x10) != 0,
        *a3 = (*(_BYTE *)(a1 + 196) & 8) != 0,
        v4 = (*(_BYTE *)(a1 + 196) & 4) != 0,
        *a4 = v4,
        !*(_DWORD *)(a1 + 160)) )
  {
    *a4 = 0;
    *a3 = 0;
    *a2 = 0;
    return;
  }
  if ( unk_4D04630 || !(unk_4D045AC | HIDWORD(qword_4D045BC)) )
    return;
  if ( (*(_BYTE *)(a1 + 198) & 0x20) != 0 && *(_QWORD *)(a1 + 128) )
    a1 = *(_QWORD *)(a1 + 128);
  if ( (*(_QWORD *)(a1 + 192) & 0x240000000LL) != 0 )
    goto LABEL_16;
  if ( (*(_BYTE *)(a1 + 195) & 3) == 1 )
  {
    *a4 = 1;
    if ( *(char *)(a1 + 192) >= 0 )
      goto LABEL_21;
    goto LABEL_23;
  }
  if ( *(char *)(a1 + 192) < 0 )
  {
LABEL_23:
    *a4 = 1;
    goto LABEL_21;
  }
  if ( !v4 )
  {
    if ( (*(_BYTE *)(a1 + 205) & 2) == 0 )
      return;
LABEL_16:
    *a3 = 1;
    if ( unk_4D045A4 && *a4 )
      goto LABEL_18;
    return;
  }
LABEL_21:
  if ( unk_4D045A4 )
  {
LABEL_18:
    *a4 = 0;
    *a3 = 1;
  }
}
