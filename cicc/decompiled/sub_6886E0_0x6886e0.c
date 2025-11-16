// Function: sub_6886E0
// Address: 0x6886e0
//
__int64 sub_6886E0()
{
  unsigned __int16 v0; // r12
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx
  unsigned int v4; // r15d
  unsigned __int16 v5; // r14

  switch ( word_4F06418[0] )
  {
    case '!':
      v0 = 64;
      goto LABEL_3;
    case '"':
      v0 = 57;
      goto LABEL_3;
    case '#':
      v0 = 60;
      goto LABEL_3;
    case '$':
      v0 = 61;
      goto LABEL_3;
    case '\'':
      v0 = 58;
      goto LABEL_3;
    case '(':
      v0 = 59;
      goto LABEL_3;
    case ')':
      v0 = 62;
      goto LABEL_3;
    case '*':
      v0 = 63;
      goto LABEL_3;
    case '2':
      v0 = 65;
      goto LABEL_3;
    case '3':
      v0 = 66;
LABEL_3:
      result = sub_7BE840(0, 0);
      if ( (_WORD)result == 56 )
      {
        v4 = dword_4F063F8;
        v5 = word_4F063FC[0];
        result = sub_7B8B50(0, 0, v2, v3);
        word_4F06418[0] = v0;
        dword_4F063F8 = v4;
        word_4F063FC[0] = v5;
      }
      break;
    default:
      result = (unsigned int)word_4F06418[0] - 33;
      break;
  }
  return result;
}
