// Function: sub_64E7D0
// Address: 0x64e7d0
//
__int64 sub_64E7D0()
{
  __int64 v0; // rbx
  char v1; // al
  __int64 v3; // rdx
  unsigned __int8 v4; // bl
  bool v5; // r12
  char v6; // al
  __int64 v7; // rcx

  v0 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
  v1 = *(_BYTE *)(v0 + 4);
  if ( dword_4F077BC && v1 == 8 )
  {
    v6 = *(_BYTE *)(v0 - 772);
    v0 -= 776;
    if ( v6 != 6 )
      return 0;
  }
  else if ( v1 != 6 )
  {
    return 0;
  }
  if ( dword_4F077C4 == 2 )
  {
    if ( (word_4F06418[0] != 1 || (unk_4D04A11 & 2) == 0) && !(unsigned int)sub_7C0F00(1, 0) )
      return 0;
  }
  else if ( word_4F06418[0] != 1 )
  {
    return 0;
  }
  if ( (unk_4D04A10 & 1) == 0 )
    return 0;
  v3 = *(_QWORD *)(v0 + 208);
  if ( (unk_4D04A12 & 2) != 0 )
  {
    if ( xmmword_4D04A20.m128i_i64[0] != v3 )
    {
      if ( !v3 )
        return 0;
      if ( !xmmword_4D04A20.m128i_i64[0] )
        return 0;
      if ( !dword_4F07588 )
        return 0;
      v7 = *(_QWORD *)(xmmword_4D04A20.m128i_i64[0] + 32);
      if ( *(_QWORD *)(v3 + 32) != v7 || !v7 )
        return 0;
    }
  }
  else if ( v3 )
  {
    return 0;
  }
  if ( (unk_4D04A10 & 2) != 0 )
    return 0;
  if ( dword_4D04964 )
  {
    v4 = byte_4F07472[0];
    v5 = byte_4F07472[0] != 3;
  }
  else
  {
    v5 = 1;
    v4 = 5;
  }
  sub_878790(&qword_4D04A00);
  if ( qword_4D0495C || !v5 )
    return 1;
  sub_684AC0(v4, 427);
  return 1;
}
