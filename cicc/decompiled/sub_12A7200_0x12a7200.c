// Function: sub_12A7200
// Address: 0x12a7200
//
__int64 __fastcall sub_12A7200(__int64 a1, int a2, int a3)
{
  __int64 result; // rax

  if ( a3 == 3 )
  {
    if ( a2 == 4 )
    {
      return sub_16432A0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
    }
    else
    {
      if ( a2 != 8 )
        sub_127B630("unexpected size3", 0);
      return sub_16432B0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
    }
  }
  else
  {
    switch ( a2 )
    {
      case 1:
        result = sub_1643330(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
        break;
      case 2:
        result = sub_1643340(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
        break;
      case 4:
        result = sub_1643350(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
        break;
      case 8:
        result = sub_1643360(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
        break;
      case 16:
        result = sub_1643370(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
        break;
      default:
        sub_127B630("unexpected size4", 0);
    }
  }
  return result;
}
