// Function: sub_87F050
// Address: 0x87f050
//
_QWORD *__fastcall sub_87F050(__int64 *a1)
{
  _QWORD *result; // rax
  _QWORD *v2; // rax
  _QWORD *v3; // [rsp-10h] [rbp-10h]

  result = *(_QWORD **)(*a1 + 32);
  if ( result )
  {
    while ( *((_BYTE *)result + 80) != 13 || *((_DWORD *)result + 10) != -1 )
    {
      result = (_QWORD *)result[1];
      if ( !result )
        goto LABEL_7;
    }
  }
  else
  {
LABEL_7:
    v2 = sub_87EBB0(0xDu, *a1, a1 + 1);
    *((_BYTE *)v2 + 81) |= 0x20u;
    v3 = v2;
    sub_879210(v2);
    return v3;
  }
  return result;
}
