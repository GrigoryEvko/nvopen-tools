// Function: sub_948510
// Address: 0x948510
//
__int64 __fastcall sub_948510(__int64 a1, int a2, int a3)
{
  __int64 result; // rax

  if ( a3 == 3 )
  {
    if ( a2 == 4 )
    {
      return sub_BCB160(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
    }
    else
    {
      if ( a2 != 8 )
        sub_91B980("unexpected size3", 0);
      return sub_BCB170(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
    }
  }
  else
  {
    switch ( a2 )
    {
      case 1:
        result = sub_BCB2B0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
        break;
      case 2:
        result = sub_BCB2C0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
        break;
      case 4:
        result = sub_BCB2D0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
        break;
      case 8:
        result = sub_BCB2E0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
        break;
      case 16:
        result = sub_BCB2F0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
        break;
      default:
        sub_91B980("unexpected size4", 0);
    }
  }
  return result;
}
