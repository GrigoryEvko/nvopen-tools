// Function: sub_691DE0
// Address: 0x691de0
//
__int64 __fastcall sub_691DE0(__int64 a1)
{
  if ( (_BYTE)a1 == 32 )
    return 45;
  if ( (unsigned __int8)a1 > 0x20u )
  {
    if ( (_BYTE)a1 != 33 )
      goto LABEL_10;
    return 46;
  }
  else
  {
    if ( (_BYTE)a1 != 16 )
    {
      if ( (_BYTE)a1 == 17 )
        return 44;
LABEL_10:
      sub_721090(a1);
    }
    return 43;
  }
}
