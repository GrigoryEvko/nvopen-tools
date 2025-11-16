// Function: sub_827590
// Address: 0x827590
//
__int64 __fastcall sub_827590(__int64 a1, char a2)
{
  __int64 result; // rax
  int v3; // r8d
  __int64 v4; // rax

  switch ( a2 )
  {
    case 'A':
      result = sub_8D2D80(a1);
      break;
    case 'B':
    case 'b':
      result = (unsigned int)sub_8D2D50(a1) || (unsigned int)sub_8D2E30(a1) || (unsigned int)sub_8D3D10(a1) != 0;
      break;
    case 'C':
      result = sub_8D3A70(a1);
      break;
    case 'D':
    case 'I':
    case 'i':
      result = sub_8D2960(a1);
      break;
    case 'E':
      result = sub_8D2870(a1);
      break;
    case 'F':
      result = sub_8D2E30(a1);
      if ( (_DWORD)result )
      {
        v4 = sub_8D46C0(a1);
        result = (unsigned int)sub_8D2310(v4) != 0;
      }
      break;
    case 'M':
      result = sub_8D3D10(a1);
      break;
    case 'N':
      result = sub_8D2660(a1);
      break;
    case 'O':
      result = sub_8D2EB0(a1);
      break;
    case 'P':
      result = sub_8D2E30(a1);
      break;
    case 'S':
      result = sub_8D28B0(a1);
      break;
    case 'a':
      result = sub_8D2DD0(a1);
      break;
    case 'n':
      v3 = sub_8D29A0(a1);
      result = 0;
      if ( !v3 )
        result = (unsigned int)sub_8D2DD0(a1) != 0;
      break;
    default:
      sub_721090();
  }
  return result;
}
