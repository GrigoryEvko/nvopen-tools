// Function: sub_729EA0
// Address: 0x729ea0
//
__int64 sub_729EA0()
{
  __int64 v0; // r8
  __int64 v2; // rax

  v0 = unk_4F07280;
  if ( qword_4F07AE8 || !unk_4F07280 )
  {
    if ( qword_4F07AE8 )
      return qword_4F07AE8;
    return v0;
  }
  v2 = *(_QWORD *)(unk_4F07280 + 40LL);
  if ( !v2 )
    return v0;
  while ( (*(_BYTE *)(v2 + 72) & 0x10) != 0 )
  {
    v2 = *(_QWORD *)(v2 + 56);
    if ( !v2 )
      return v0;
  }
  if ( !*(_QWORD *)(v2 + 8) && *(_DWORD *)(v2 + 24) == 2 && *(_DWORD *)(v2 + 32) == 1 )
  {
    qword_4F07AE8 = v2;
    return v2;
  }
  qword_4F07AE8 = unk_4F07280;
  return unk_4F07280;
}
