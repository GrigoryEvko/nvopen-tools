// Function: sub_7AFE70
// Address: 0x7afe70
//
__int64 sub_7AFE70()
{
  unsigned int v0; // r8d

  v0 = 2;
  if ( !unk_4F064B0 )
    return v0;
  if ( (*(_BYTE *)(unk_4F064B0 + 88LL) & 1) == 0 )
    return v0;
  v0 = *(unsigned __int8 *)(unk_4F064B0 + 89LL);
  if ( (unsigned __int8)v0 > 1u || !unk_4F064B8 )
    return v0;
  *(_BYTE *)(unk_4F064B0 + 89LL) = 2;
  return 2;
}
