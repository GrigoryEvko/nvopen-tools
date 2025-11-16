// Function: sub_A56760
// Address: 0xa56760
//
_BYTE *__fastcall sub_A56760(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // r14
  size_t v3; // rdx
  size_t v4; // r13
  _BYTE *v5; // rax
  void *v6; // rdx
  _BYTE *result; // rax

  v2 = (unsigned __int8 *)sub_AA8810(a1);
  v4 = v3;
  v5 = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)v5 >= *(_QWORD *)(a2 + 24) )
  {
    sub_CB5D20(a2, 36);
  }
  else
  {
    *(_QWORD *)(a2 + 32) = v5 + 1;
    *v5 = 36;
  }
  sub_A54F00(a2, v2, v4);
  v6 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v6 <= 9u )
  {
    sub_CB6200(a2, " = comdat ", 10);
  }
  else
  {
    qmemcpy(v6, " = comdat ", 10);
    *(_QWORD *)(a2 + 32) += 10LL;
  }
  switch ( *(_DWORD *)(a1 + 8) )
  {
    case 0:
      sub_904010(a2, "any");
      break;
    case 1:
      sub_904010(a2, "exactmatch");
      break;
    case 2:
      sub_904010(a2, "largest");
      break;
    case 3:
      sub_904010(a2, "nodeduplicate");
      break;
    case 4:
      sub_904010(a2, "samesize");
      break;
    default:
      break;
  }
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 10);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 10;
  return result;
}
