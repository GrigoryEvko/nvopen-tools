// Function: sub_7AC2A0
// Address: 0x7ac2a0
//
__int64 sub_7AC2A0()
{
  __int64 result; // rax
  unsigned __int8 v1; // di

  result = *(unsigned __int8 *)(xmmword_4F06380[0].m128i_i64[0] + 160);
  if ( dword_4F077C4 == 2 )
  {
    if ( unk_4F07778 <= 201102 && !dword_4F07774 )
      goto LABEL_3;
  }
  else if ( unk_4F07778 <= 199900 )
  {
LABEL_3:
    if ( (unsigned __int8)(result - 7) > 3u )
    {
      result = (__int64)sub_72BA30((byte_4B6DF90[result] == 0) + 7);
      xmmword_4F06380[0].m128i_i64[0] = result;
    }
    return result;
  }
  if ( unk_4F06AC9 != (_BYTE)result )
  {
    v1 = unk_4F06AC8;
    if ( unk_4F06AC8 != (_BYTE)result )
    {
      if ( byte_4B6DF90[result] )
        v1 = unk_4F06AC9;
      result = (__int64)sub_72BA30(v1);
      xmmword_4F06380[0].m128i_i64[0] = result;
    }
  }
  return result;
}
