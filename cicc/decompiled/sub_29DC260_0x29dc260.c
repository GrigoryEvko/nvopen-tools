// Function: sub_29DC260
// Address: 0x29dc260
//
__int64 __fastcall sub_29DC260(__int64 a1, _BYTE *a2, char a3)
{
  __int64 result; // rax

  result = a2[32] & 0xF;
  if ( *(_BYTE *)(a1 + 24) )
  {
    if ( (unsigned int)(result - 7) <= 1 )
    {
      if ( a3 )
        return 0;
    }
  }
  else if ( *(_QWORD *)(a1 + 16) )
  {
    switch ( a2[32] & 0xF )
    {
      case 0:
      case 3:
        if ( !(unsigned __int8)sub_29DBA40(a1, (__int64)a2) )
          return a2[32] & 0xF;
        result = 1;
        if ( *a2 == 1 )
          return a2[32] & 0xF;
        return result;
      case 1:
        if ( (unsigned __int8)sub_29DBA40(a1, (__int64)a2) )
          return a2[32] & 0xF;
        else
          return 0;
      case 2:
      case 4:
        return result;
      case 5:
        goto LABEL_8;
      case 6:
        return 6;
      case 7:
      case 8:
        if ( a3 )
        {
LABEL_8:
          if ( (unsigned __int8)sub_29DBA40(a1, (__int64)a2) )
            result = *a2 != 1;
          else
            result = 0;
        }
        break;
      case 9:
        result = 9;
        break;
      case 0xA:
        result = 10;
        break;
      default:
        BUG();
    }
  }
  return result;
}
