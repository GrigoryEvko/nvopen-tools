// Function: sub_5CCA00
// Address: 0x5cca00
//
_DWORD *sub_5CCA00()
{
  _DWORD *result; // rax

  result = (_DWORD *)word_4F06418[0];
LABEL_2:
  if ( (_WORD)result == 25 )
    goto LABEL_9;
LABEL_3:
  while ( (_WORD)result == 142 )
  {
    result = &dword_4D043E0;
    if ( !dword_4D043E0 )
      return result;
LABEL_6:
    sub_7B8B50();
    result = (_DWORD *)word_4F06418[0];
    if ( word_4F06418[0] != 27 )
      goto LABEL_2;
    sub_7BDC20(0);
    result = (_DWORD *)word_4F06418[0];
    if ( word_4F06418[0] != 28 )
      goto LABEL_2;
    while ( 1 )
    {
      sub_7B8B50();
      result = (_DWORD *)word_4F06418[0];
      if ( word_4F06418[0] != 25 )
        break;
LABEL_9:
      result = &dword_4D043F8;
      if ( !dword_4D043F8 )
        return result;
      if ( (unsigned __int16)sub_7BE840(0, 0) != 25 )
      {
        result = (_DWORD *)word_4F06418[0];
        goto LABEL_3;
      }
      sub_7BDC20(0);
      result = (_DWORD *)word_4F06418[0];
      if ( word_4F06418[0] != 26 )
        goto LABEL_2;
    }
  }
  if ( (_WORD)result != 132 )
  {
    if ( (_WORD)result != 248 )
      return result;
    goto LABEL_6;
  }
  result = (_DWORD *)dword_4D043DC;
  if ( dword_4D043DC )
    goto LABEL_6;
  return result;
}
