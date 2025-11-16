// Function: sub_B2FC00
// Address: 0xb2fc00
//
__int64 __fastcall sub_B2FC00(_BYTE *a1)
{
  switch ( a1[32] & 0xF )
  {
    case 0:
    case 2:
    case 4:
    case 6:
    case 7:
    case 8:
    case 9:
    case 0xA:
      goto LABEL_3;
    case 1:
    case 3:
    case 5:
      if ( (_BYTE)qword_4F816A8 )
      {
        if ( *a1 || !(unsigned __int8)sub_CE9220() || (unsigned __int8)sub_B2F6B0((__int64)a1) )
          return 1;
      }
      else
      {
LABEL_3:
        if ( (unsigned __int8)sub_B2F6B0((__int64)a1) )
          return 1;
      }
      return sub_B2FBD0((__int64)a1);
    default:
      BUG();
  }
}
