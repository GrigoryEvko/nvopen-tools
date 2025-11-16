// Function: sub_B30810
// Address: 0xb30810
//
__int64 __fastcall sub_B30810(_QWORD *a1)
{
  unsigned __int8 v1; // al

  v1 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 2 )
    return sub_B307B0(a1);
  if ( v1 > 2u )
  {
    if ( v1 != 3 )
      BUG();
    return sub_B30290((__int64)a1);
  }
  else if ( v1 )
  {
    return sub_B30340(a1);
  }
  else
  {
    return sub_B2E860(a1);
  }
}
