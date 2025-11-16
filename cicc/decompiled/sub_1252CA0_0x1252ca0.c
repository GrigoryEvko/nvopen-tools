// Function: sub_1252CA0
// Address: 0x1252ca0
//
_BYTE *__fastcall sub_1252CA0(__int64 a1, char a2, unsigned __int64 a3, char a4)
{
  char v4; // bl
  _BYTE *v5; // rax
  char *v6; // rax
  _BYTE *result; // rax

  v4 = a4 | (16 * a2);
  if ( a3 > 0x50 )
    v4 |= 1u;
  v5 = *(_BYTE **)(a1 + 32);
  if ( (unsigned __int64)v5 < *(_QWORD *)(a1 + 24) )
  {
    *(_QWORD *)(a1 + 32) = v5 + 1;
    *v5 = 3;
    v6 = *(char **)(a1 + 32);
    if ( (unsigned __int64)v6 < *(_QWORD *)(a1 + 24) )
      goto LABEL_5;
LABEL_8:
    a1 = sub_CB5D20(a1, v4);
    result = *(_BYTE **)(a1 + 32);
    if ( (unsigned __int64)result < *(_QWORD *)(a1 + 24) )
      goto LABEL_6;
    return (_BYTE *)sub_CB5D20(a1, 0);
  }
  a1 = sub_CB5D20(a1, 3);
  v6 = *(char **)(a1 + 32);
  if ( (unsigned __int64)v6 >= *(_QWORD *)(a1 + 24) )
    goto LABEL_8;
LABEL_5:
  *(_QWORD *)(a1 + 32) = v6 + 1;
  *v6 = v4;
  result = *(_BYTE **)(a1 + 32);
  if ( (unsigned __int64)result < *(_QWORD *)(a1 + 24) )
  {
LABEL_6:
    *(_QWORD *)(a1 + 32) = result + 1;
    *result = 0;
    return result;
  }
  return (_BYTE *)sub_CB5D20(a1, 0);
}
