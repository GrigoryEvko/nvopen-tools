// Function: sub_2537B10
// Address: 0x2537b10
//
__int64 __fastcall sub_2537B10(__int64 *a1, __int64 a2)
{
  unsigned __int8 **v2; // rax
  unsigned __int8 *v4; // rax
  unsigned __int64 v5; // r12

  v2 = *(unsigned __int8 ***)(a2 + 24);
  if ( *(_BYTE *)v2 != 62 )
    return 0;
  v4 = sub_BD3990(*(v2 - 4), a2);
  v5 = (unsigned __int64)v4;
  if ( *v4 == 60 || (unsigned __int8)sub_CF6FD0(v4) )
    return sub_252BB70(*a1, a1[1], v5, 1);
  else
    return 0;
}
