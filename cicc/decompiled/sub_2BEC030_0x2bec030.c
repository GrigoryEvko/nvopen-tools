// Function: sub_2BEC030
// Address: 0x2bec030
//
__int64 __fastcall sub_2BEC030(_DWORD *a1)
{
  int v1; // eax
  char v3; // si
  int v4; // edx

  v1 = a1[38];
  if ( v1 == 10 )
  {
    if ( (unsigned __int8)sub_2BE0030((__int64)a1) )
    {
      v3 = 1;
      goto LABEL_5;
    }
    v1 = a1[38];
  }
  if ( v1 != 9 )
    return 0;
  v3 = 0;
  if ( !(unsigned __int8)sub_2BE0030((__int64)a1) )
    return 0;
LABEL_5:
  v4 = *a1 & 8;
  if ( (*a1 & 1) != 0 )
  {
    if ( v4 )
      sub_2BEA5B0((__int64)a1, v3);
    else
      sub_2BE8F10((__int64)a1, v3);
    return 1;
  }
  else
  {
    if ( v4 )
      sub_2BEB870((__int64)a1, v3);
    else
      sub_2BE8220((__int64)a1, v3);
    return 1;
  }
}
