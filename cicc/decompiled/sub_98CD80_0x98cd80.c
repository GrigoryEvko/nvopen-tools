// Function: sub_98CD80
// Address: 0x98cd80
//
__int64 __fastcall sub_98CD80(char *a1)
{
  unsigned int v1; // r12d
  char v2; // dl
  __int64 v4; // rax
  __int64 v5; // rax

  v2 = *a1;
  LOBYTE(v1) = v2 == 30 || v2 == 36;
  if ( (_BYTE)v1 )
    return 0;
  if ( v2 != 81 )
  {
    if ( !(unsigned __int8)sub_B46790(a1, 0) )
      return sub_B46900(a1);
    return v1;
  }
  v4 = sub_B43CB0(a1);
  v5 = sub_B2E500(v4);
  LOBYTE(v1) = (unsigned int)sub_B2A630(v5) == 10;
  return v1;
}
