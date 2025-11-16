// Function: sub_5E4740
// Address: 0x5e4740
//
__int64 sub_5E4740()
{
  __int64 v0; // rax
  char v1; // dl
  unsigned int v2; // r8d
  __int64 v4; // rdx

  v0 = unk_4F04C68 + 776LL * unk_4F04C64;
  v1 = *(_BYTE *)(v0 + 4);
  if ( v1 != 8 )
    goto LABEL_5;
  do
  {
    v1 = *(_BYTE *)(v0 - 772);
    v0 -= 776;
  }
  while ( v1 == 8 );
  if ( v1 != 6 )
    goto LABEL_8;
  do
  {
    v1 = *(_BYTE *)(v0 - 772);
    v0 -= 776;
LABEL_5:
    ;
  }
  while ( v1 == 6 );
  while ( v1 == 2 )
  {
    v1 = *(_BYTE *)(v0 - 772);
    v0 -= 776;
LABEL_8:
    ;
  }
  if ( v1 != 17 )
  {
    v2 = 1;
    if ( ((v1 - 8) & 0xF7) != 0 )
    {
      v2 = 0;
      if ( v1 == 1 )
      {
        v2 = 1;
        if ( unk_4F077BC )
        {
          if ( qword_4F077A8 <= 0x76BFu )
            return unk_4D0448C != 0;
        }
      }
    }
    return v2;
  }
  v2 = 0;
  if ( (*(_BYTE *)(*(_QWORD *)(v0 + 216) + 206LL) & 2) == 0 )
    return v2;
  v4 = v0 - 1552;
  if ( *(_BYTE *)(v0 - 772) != 9 )
    v4 = v0 - 776;
  return (*(_BYTE *)(*(_QWORD *)(**(_QWORD **)(v4 + 208) + 96LL) + 181LL) & 4) != 0;
}
