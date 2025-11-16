// Function: sub_724DC0
// Address: 0x724dc0
//
_QWORD *sub_724DC0()
{
  __int64 v0; // r8
  char v1; // al
  char v2; // al

  v0 = unk_4F06BB8;
  if ( !unk_4F06BB8 )
    return sub_7247C0(208);
  unk_4F06BB8 = *(_QWORD *)(unk_4F06BB8 + 120LL);
  v1 = *(_BYTE *)(v0 - 8) | 1;
  *(_BYTE *)(v0 - 8) = v1;
  v2 = (2 * (unk_4D03FE8 == 0)) | v1 & 0xFD;
  *(_BYTE *)(v0 - 8) = v2 & 0x8B;
  *(_BYTE *)(v0 - 8) = (8 * (unk_4F06CFC & 1)) | v2 & 3;
  return (_QWORD *)v0;
}
