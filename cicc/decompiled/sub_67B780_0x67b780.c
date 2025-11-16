// Function: sub_67B780
// Address: 0x67b780
//
void __fastcall sub_67B780(__int64 a1, _QWORD *a2)
{
  switch ( (char)a1 )
  {
    case 2:
      return;
    case 4:
      ++*a2;
      break;
    case 5:
    case 6:
      ++a2[1];
      break;
    case 7:
    case 8:
      ++a2[2];
      break;
    case 9:
    case 10:
    case 11:
      ++a2[3];
      break;
    default:
      sub_721090(a1);
  }
}
