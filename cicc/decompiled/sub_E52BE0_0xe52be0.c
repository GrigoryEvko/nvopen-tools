// Function: sub_E52BE0
// Address: 0xe52be0
//
_BYTE *__fastcall sub_E52BE0(__int64 a1, int a2)
{
  _BYTE *result; // rax
  __int64 v4; // rdi

  result = *(_BYTE **)(a1 + 312);
  if ( result[185] )
  {
    v4 = *(_QWORD *)(a1 + 304);
    switch ( a2 )
    {
      case 0:
        sub_904010(v4, "\t.data_region");
        result = sub_E4D880(a1);
        break;
      case 1:
        sub_904010(v4, "\t.data_region jt8");
        result = sub_E4D880(a1);
        break;
      case 2:
        sub_904010(v4, "\t.data_region jt16");
        result = sub_E4D880(a1);
        break;
      case 3:
        sub_904010(v4, "\t.data_region jt32");
        result = sub_E4D880(a1);
        break;
      case 4:
        sub_904010(v4, "\t.end_data_region");
        goto LABEL_4;
      default:
LABEL_4:
        result = sub_E4D880(a1);
        break;
    }
  }
  return result;
}
