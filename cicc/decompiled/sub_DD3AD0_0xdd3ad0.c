// Function: sub_DD3AD0
// Address: 0xdd3ad0
//
_QWORD *__fastcall sub_DD3AD0(__int64 a1, unsigned __int16 a2, __int64 a3, __int64 a4)
{
  if ( a2 != 4 )
  {
    if ( a2 > 4u )
    {
      if ( a2 == 14 )
        return sub_DD3A70(a1, a3, a4);
    }
    else
    {
      if ( a2 == 2 )
        return sub_DC5200(a1, a3, a4, 0);
      if ( a2 == 3 )
        return sub_DC2B70(a1, a3, a4, 0);
    }
    BUG();
  }
  return sub_DC5000(a1, a3, a4, 0);
}
