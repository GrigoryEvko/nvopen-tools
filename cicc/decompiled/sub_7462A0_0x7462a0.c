// Function: sub_7462A0
// Address: 0x7462a0
//
char *__fastcall sub_7462A0(char a1, int a2)
{
  char *v2; // r12

  v2 = "char";
  if ( byte_4F068B0[0] != a1 )
  {
    switch ( a1 )
    {
      case 0:
        v2 = "char";
        break;
      case 1:
        v2 = "signed char";
        break;
      case 2:
        v2 = "unsigned char";
        break;
      case 3:
        v2 = "short";
        break;
      case 4:
        v2 = "unsigned short";
        break;
      case 5:
        v2 = "int";
        break;
      case 6:
        v2 = "unsigned int";
        break;
      case 7:
        v2 = "long";
        break;
      case 8:
        v2 = "unsigned long";
        break;
      case 9:
        v2 = "long long";
        goto LABEL_4;
      case 10:
        v2 = "unsigned long long";
LABEL_4:
        if ( a2 )
        {
          if ( !sub_5D76E0() )
          {
            if ( unk_4F068C0 )
            {
              v2 = "__int64";
              if ( a1 != 9 )
                v2 = "unsigned __int64";
            }
          }
        }
        break;
      case 11:
        v2 = "**128-BIT SIGNED INTEGER**";
        if ( HIDWORD(qword_4F077B4) )
        {
          v2 = "__int128";
          if ( !unk_4F068D0 )
          {
            v2 = "__int128_t";
            if ( unk_4F068E0 )
            {
              if ( qword_4F068D8 >= 0x9E98u )
                v2 = "__int128";
            }
          }
          if ( a2 && sub_5D76E0() )
            v2 = "long long";
        }
        break;
      case 12:
        v2 = "**128-BIT UNSIGNED INTEGER**";
        if ( HIDWORD(qword_4F077B4) )
        {
          v2 = "unsigned __int128";
          if ( !unk_4F068D0 )
          {
            v2 = "__uint128_t";
            if ( unk_4F068E0 )
            {
              if ( qword_4F068D8 >= 0x9E98u )
                v2 = "unsigned __int128";
            }
          }
          if ( a2 && sub_5D76E0() )
            v2 = "unsigned long long";
        }
        break;
      default:
        v2 = "**BAD-INT-KIND**";
        break;
    }
  }
  return v2;
}
