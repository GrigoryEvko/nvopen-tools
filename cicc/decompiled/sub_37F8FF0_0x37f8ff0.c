// Function: sub_37F8FF0
// Address: 0x37f8ff0
//
bool __fastcall sub_37F8FF0(__int64 *a1, __int64 a2)
{
  bool result; // al
  __int64 v3; // r12
  unsigned __int8 *v4; // rax
  unsigned __int8 *v5; // rdi

  result = 1;
  if ( a1[54] == a1[55] )
  {
    v3 = *a1;
    if ( (*(_BYTE *)(*a1 + 2) & 8) != 0
      && ((unsigned int)sub_A746B0((_QWORD *)(v3 + 120))
       || (a2 = 41, !(unsigned __int8)sub_B2D610(v3, 41))
       || (*(_BYTE *)(v3 + 2) & 8) != 0) )
    {
      v4 = (unsigned __int8 *)sub_B2E500(v3);
      v5 = sub_BD3990(v4, a2);
      if ( *v5 >= 4u )
        v5 = 0;
      return (unsigned int)sub_B2A630((__int64)v5) == 0;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
