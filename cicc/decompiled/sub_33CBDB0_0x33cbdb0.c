// Function: sub_33CBDB0
// Address: 0x33cbdb0
//
__int64 __fastcall sub_33CBDB0(unsigned int a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  _QWORD v5[4]; // [rsp+0h] [rbp-20h] BYREF

  v5[0] = a3;
  v5[1] = a4;
  if ( !(_WORD)a3 )
  {
    if ( sub_3007070((__int64)v5) )
      goto LABEL_5;
LABEL_16:
    result = a1 | a2;
    if ( (unsigned int)result > 0x17 )
      return (a1 | a2) & 0xFFFFFFEF;
    return result;
  }
  if ( (unsigned __int16)(a3 - 2) > 7u && (unsigned __int16)(a3 - 17) > 0x6Cu && (unsigned __int16)(a3 - 176) > 0x1Fu )
    goto LABEL_16;
LABEL_5:
  if ( a1 == 17 )
    goto LABEL_29;
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
        {
LABEL_11:
          result = a1 | a2;
          goto LABEL_12;
        }
LABEL_23:
        if ( a2 == 22 )
          goto LABEL_24;
LABEL_38:
        BUG();
      }
      goto LABEL_24;
    }
    if ( a1 != 22 )
      goto LABEL_38;
LABEL_29:
    if ( a2 <= 0xD )
    {
      if ( a2 > 9 )
        goto LABEL_24;
    }
    else if ( a2 - 17 <= 5 )
    {
      goto LABEL_24;
    }
    goto LABEL_38;
  }
  if ( a1 - 10 > 3 )
    goto LABEL_38;
  if ( a2 != 17 )
  {
    if ( a2 <= 0x11 )
    {
      if ( a2 - 10 <= 3 )
        goto LABEL_11;
      goto LABEL_38;
    }
    result = 24;
    if ( a2 <= 0x15 )
      return result;
    goto LABEL_23;
  }
LABEL_24:
  result = a2 | a1;
  if ( (unsigned int)result <= 0x17 )
  {
LABEL_12:
    if ( (_DWORD)result == 14 )
      return 22;
    return result;
  }
  result = (unsigned int)result & 0xFFFFFFEF;
  if ( (_DWORD)result == 14 )
    return 22;
  return result;
}
