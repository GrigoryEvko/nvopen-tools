// Function: sub_33CBF10
// Address: 0x33cbf10
//
__int64 __fastcall sub_33CBF10(unsigned int a1, unsigned int a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // ebx
  __int64 result; // rax
  _QWORD v6[4]; // [rsp+0h] [rbp-20h] BYREF

  v6[0] = a3;
  v6[1] = a4;
  if ( !(_WORD)a3 )
  {
    if ( sub_3007070((__int64)v6) )
      goto LABEL_5;
    return a2 & a1;
  }
  if ( (unsigned __int16)(a3 - 2) > 7u && (unsigned __int16)(a3 - 17) > 0x6Cu && (unsigned __int16)(a3 - 176) > 0x1Fu )
    return a2 & a1;
LABEL_5:
  if ( a1 == 17 )
    goto LABEL_23;
  if ( a1 > 0x11 )
  {
    if ( a1 <= 0x15 )
    {
      if ( a2 != 17 )
      {
        if ( a2 <= 0x11 )
        {
          if ( a2 - 10 <= 3 )
            return 24;
          goto LABEL_38;
        }
        if ( a2 <= 0x15 )
          return a1 & a2;
        if ( a2 != 22 )
LABEL_38:
          BUG();
      }
      goto LABEL_21;
    }
    if ( a1 != 22 )
      goto LABEL_38;
LABEL_23:
    if ( a2 <= 0xD )
    {
      if ( a2 <= 9 )
        goto LABEL_38;
    }
    else if ( a2 - 17 > 5 )
    {
      goto LABEL_38;
    }
    goto LABEL_21;
  }
  if ( a1 - 10 > 3 )
    goto LABEL_38;
  if ( a2 != 17 )
  {
    if ( a2 <= 0x11 )
    {
      if ( a2 - 10 > 3 )
        goto LABEL_38;
    }
    else
    {
      result = 24;
      if ( a2 <= 0x15 )
        return result;
      if ( a2 != 22 )
        goto LABEL_38;
    }
  }
LABEL_21:
  v4 = a1 & a2;
  switch ( a1 & a2 )
  {
    case 1u:
    case 9u:
      result = 17;
      break;
    case 2u:
      result = 10;
      break;
    case 4u:
      result = 12;
      break;
    case 8u:
      result = 0;
      break;
    default:
      return v4;
  }
  return result;
}
