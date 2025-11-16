// Function: sub_E52CD0
// Address: 0xe52cd0
//
_BYTE *__fastcall sub_E52CD0(__int64 a1, int a2)
{
  __int64 v3; // rdi
  _BYTE *v4; // rax
  _BYTE *result; // rax
  _BYTE *v6; // rax
  _BYTE *v7; // rax

  v3 = *(_QWORD *)(a1 + 304);
  switch ( a2 )
  {
    case 0:
      sub_904010(v3, "\t.syntax unified");
      result = sub_E4D880(a1);
      break;
    case 1:
      sub_904010(v3, ".subsections_via_symbols");
      result = sub_E4D880(a1);
      break;
    case 2:
      v7 = *(_BYTE **)(v3 + 32);
      if ( (unsigned __int64)v7 >= *(_QWORD *)(v3 + 24) )
      {
        v3 = sub_CB5D20(v3, 9);
      }
      else
      {
        *(_QWORD *)(v3 + 32) = v7 + 1;
        *v7 = 9;
      }
      sub_904010(v3, *(const char **)(*(_QWORD *)(a1 + 312) + 152LL));
      result = sub_E4D880(a1);
      break;
    case 3:
      v6 = *(_BYTE **)(v3 + 32);
      if ( (unsigned __int64)v6 >= *(_QWORD *)(v3 + 24) )
      {
        v3 = sub_CB5D20(v3, 9);
      }
      else
      {
        *(_QWORD *)(v3 + 32) = v6 + 1;
        *v6 = 9;
      }
      sub_904010(v3, *(const char **)(*(_QWORD *)(a1 + 312) + 160LL));
      result = sub_E4D880(a1);
      break;
    case 4:
      v4 = *(_BYTE **)(v3 + 32);
      if ( (unsigned __int64)v4 >= *(_QWORD *)(v3 + 24) )
      {
        v3 = sub_CB5D20(v3, 9);
      }
      else
      {
        *(_QWORD *)(v3 + 32) = v4 + 1;
        *v4 = 9;
      }
      sub_904010(v3, *(const char **)(*(_QWORD *)(a1 + 312) + 168LL));
      goto LABEL_5;
    default:
LABEL_5:
      result = sub_E4D880(a1);
      break;
  }
  return result;
}
