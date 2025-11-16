// Function: sub_154C270
// Address: 0x154c270
//
_BYTE *__fastcall sub_154C270(__int64 a1, __int64 a2)
{
  const char *v2; // r14
  size_t v3; // rdx
  size_t v4; // r13
  _BYTE *v5; // rax
  void *v6; // rdx
  _BYTE *result; // rax

  v2 = (const char *)sub_1580C70(a1);
  v4 = v3;
  v5 = *(_BYTE **)(a2 + 24);
  if ( (unsigned __int64)v5 >= *(_QWORD *)(a2 + 16) )
  {
    sub_16E7DE0(a2, 36);
  }
  else
  {
    *(_QWORD *)(a2 + 24) = v5 + 1;
    *v5 = 36;
  }
  sub_154B650(a2, v2, v4);
  v6 = *(void **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v6 <= 9u )
  {
    sub_16E7EE0(a2, " = comdat ", 10);
  }
  else
  {
    qmemcpy(v6, " = comdat ", 10);
    *(_QWORD *)(a2 + 24) += 10LL;
  }
  switch ( *(_DWORD *)(a1 + 8) )
  {
    case 0:
      sub_1263B40(a2, "any");
      break;
    case 1:
      sub_1263B40(a2, "exactmatch");
      break;
    case 2:
      sub_1263B40(a2, "largest");
      break;
    case 3:
      sub_1263B40(a2, "noduplicates");
      break;
    case 4:
      sub_1263B40(a2, "samesize");
      break;
    default:
      break;
  }
  result = *(_BYTE **)(a2 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 16) )
    return (_BYTE *)sub_16E7DE0(a2, 10);
  *(_QWORD *)(a2 + 24) = result + 1;
  *result = 10;
  return result;
}
