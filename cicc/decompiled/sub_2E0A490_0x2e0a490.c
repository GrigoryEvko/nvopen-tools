// Function: sub_2E0A490
// Address: 0x2e0a490
//
_QWORD *__fastcall sub_2E0A490(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v3; // rdx
  _QWORD *v4; // r8
  signed __int64 v5; // rdx
  _QWORD *v6; // rdx

  result = *(_QWORD **)a1;
  v3 = 24LL * *(unsigned int *)(a1 + 8);
  v4 = (_QWORD *)(*(_QWORD *)a1 + v3);
  v5 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 3);
  if ( v5 >> 2 )
  {
    v6 = &result[12 * (v5 >> 2)];
    while ( a2 != result[2] )
    {
      if ( a2 == result[5] )
      {
        result += 3;
        goto LABEL_8;
      }
      if ( a2 == result[8] )
      {
        result += 6;
        goto LABEL_8;
      }
      if ( a2 == result[11] )
      {
        result += 9;
        goto LABEL_8;
      }
      result += 12;
      if ( result == v6 )
      {
        v5 = 0xAAAAAAAAAAAAAAABLL * (v4 - result);
        goto LABEL_11;
      }
    }
    goto LABEL_8;
  }
LABEL_11:
  if ( v5 == 2 )
    goto LABEL_17;
  if ( v5 == 3 )
  {
    if ( a2 == result[2] )
      goto LABEL_8;
    result += 3;
LABEL_17:
    if ( a2 == result[2] )
      goto LABEL_8;
    result += 3;
    goto LABEL_19;
  }
  if ( v5 != 1 )
    return (_QWORD *)sub_2E0A2D0(a1, a2);
LABEL_19:
  if ( a2 != result[2] )
    return (_QWORD *)sub_2E0A2D0(a1, a2);
LABEL_8:
  if ( v4 == result )
    return (_QWORD *)sub_2E0A2D0(a1, a2);
  return result;
}
