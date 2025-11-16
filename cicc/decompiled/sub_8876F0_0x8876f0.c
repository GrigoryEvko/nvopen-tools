// Function: sub_8876F0
// Address: 0x8876f0
//
__int64 sub_8876F0()
{
  unsigned int v0; // r12d
  unsigned __int64 i; // rdx
  unsigned int v3; // edx
  __int64 v4; // rax

  v0 = 0;
  if ( word_4F06418[0] == 1 )
  {
    for ( i = (unsigned __int64)qword_4D04A00 >> 3; ; LODWORD(i) = v3 + 1 )
    {
      v3 = qword_4F5FE48[1] & i;
      v4 = *qword_4F5FE48 + 16LL * v3;
      if ( qword_4D04A00 == *(_QWORD *)v4 )
        break;
      if ( !*(_QWORD *)v4 )
        return 0;
    }
    v0 = *(unsigned __int16 *)(v4 + 8);
    if ( (_WORD)v0 )
    {
      if ( !sub_7C8F50(v0, *(char **)(qword_4D04A00 + 8)) )
        sub_721090();
    }
  }
  return v0;
}
