// Function: sub_1C56FB0
// Address: 0x1c56fb0
//
_QWORD *__fastcall sub_1C56FB0(_QWORD *a1, _QWORD *a2, unsigned __int64 *a3)
{
  unsigned __int64 v4; // r14
  _QWORD *result; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx

  if ( a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_1C56EF0((__int64)a1, a3);
    v10 = a1[4];
    v11 = *(_QWORD *)(v10 + 32);
    if ( *a3 <= v11 && (*a3 != v11 || a3[1] <= *(_QWORD *)(v10 + 40)) )
      return sub_1C56EF0((__int64)a1, a3);
    return 0;
  }
  v4 = *a3;
  if ( a2[4] > *a3 )
    goto LABEL_3;
  if ( a2[4] == *a3 )
  {
    v12 = a3[1];
    v13 = a2[5];
    if ( v13 > v12 )
    {
LABEL_3:
      result = a2;
      if ( (_QWORD *)a1[3] == a2 )
        return result;
      v6 = sub_220EF80(a2);
      v7 = v6;
      if ( v4 > *(_QWORD *)(v6 + 32) || v4 == *(_QWORD *)(v6 + 32) && *(_QWORD *)(v6 + 40) < a3[1] )
      {
        result = 0;
        if ( *(_QWORD *)(v7 + 24) )
          return a2;
        return result;
      }
      return sub_1C56EF0((__int64)a1, a3);
    }
  }
  else
  {
    if ( a2[4] < *a3 )
      goto LABEL_12;
    v13 = a2[5];
    v12 = a3[1];
  }
  if ( v13 >= v12 )
    return a2;
LABEL_12:
  if ( (_QWORD *)a1[4] == a2 )
    return 0;
  v8 = sub_220EEE0(a2);
  v9 = v8;
  if ( v4 >= *(_QWORD *)(v8 + 32) && (v4 != *(_QWORD *)(v8 + 32) || a3[1] >= *(_QWORD *)(v8 + 40)) )
    return sub_1C56EF0((__int64)a1, a3);
  result = 0;
  if ( a2[3] )
    return (_QWORD *)v9;
  return result;
}
