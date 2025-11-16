// Function: sub_2208BC0
// Address: 0x2208bc0
//
_QWORD *__fastcall sub_2208BC0(_QWORD *a1, _QWORD *a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rdx
  __int64 v4; // rdx
  __int64 v5; // rax
  _QWORD *v6; // rax
  _QWORD *v7; // rdx

  result = (_QWORD *)*a1;
  v3 = (_QWORD *)*a2;
  if ( (_QWORD *)*a1 == a1 )
  {
    if ( a2 != v3 )
    {
      *result = v3;
      v7 = (_QWORD *)a2[1];
      result[1] = v7;
      *v7 = result;
      *(_QWORD *)(*result + 8LL) = result;
      a2[1] = a2;
      *a2 = a2;
    }
  }
  else if ( a2 == v3 )
  {
    *a2 = result;
    v6 = (_QWORD *)a1[1];
    a2[1] = v6;
    *v6 = a2;
    result = (_QWORD *)*a2;
    *(_QWORD *)(*a2 + 8LL) = a2;
    a1[1] = a1;
    *a1 = a1;
  }
  else
  {
    *a1 = v3;
    v4 = a2[1];
    *a2 = result;
    v5 = a1[1];
    a1[1] = v4;
    a2[1] = v5;
    *(_QWORD *)a1[1] = a1;
    *(_QWORD *)(*a1 + 8LL) = a1;
    *(_QWORD *)a2[1] = a2;
    result = (_QWORD *)*a2;
    *(_QWORD *)(*a2 + 8LL) = a2;
  }
  return result;
}
