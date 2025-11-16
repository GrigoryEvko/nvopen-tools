// Function: sub_732E60
// Address: 0x732e60
//
__int64 __fastcall sub_732E60(unsigned __int8 *a1, unsigned __int8 a2, _QWORD *a3)
{
  __int64 result; // rax
  _QWORD *v4; // rdx
  _QWORD *v5; // rcx

  result = *a1;
  a1[8] = a2;
  *((_QWORD *)a1 + 2) = a3;
  if ( (_BYTE)result != 2 )
  {
    switch ( a2 )
    {
      case 0xDu:
        v4 = a3 + 8;
        break;
      case 0x13u:
        v4 = a3 + 3;
        break;
      case 0x14u:
        v4 = a3 + 2;
        break;
      case 0x17u:
        v5 = a3 + 7;
        v4 = a3 + 11;
        if ( (_BYTE)result == 3 )
          v4 = v5;
        break;
      case 0x1Eu:
        v4 = a3 + 5;
        break;
      case 0x1Fu:
        v4 = a3 + 4;
        break;
      default:
        sub_721090();
    }
    *v4 = a1;
  }
  return result;
}
