// Function: sub_737810
// Address: 0x737810
//
void __fastcall sub_737810(unsigned __int8 a1, _DWORD *a2, _DWORD *a3)
{
  *a2 = 0;
  *a3 = 0;
  if ( a1 > 0x56u )
  {
    if ( a1 == 92 )
    {
LABEL_4:
      *a2 = 1;
      return;
    }
    if ( (unsigned __int8)(a1 - 105) <= 4u )
      sub_721090();
  }
  else
  {
    if ( a1 > 0x48u )
    {
      *a3 = 1;
      return;
    }
    if ( (unsigned __int8)(a1 - 53) <= 1u )
      goto LABEL_4;
  }
}
