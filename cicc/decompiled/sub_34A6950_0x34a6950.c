// Function: sub_34A6950
// Address: 0x34a6950
//
_QWORD *__fastcall sub_34A6950(_QWORD *a1, _QWORD *a2, unsigned __int64 *a3)
{
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  _QWORD *result; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx

  if ( a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_34A55B0((__int64)a1, a3);
    v11 = a1[4];
    if ( *(_QWORD *)(v11 + 32) >= *a3 && (*(_QWORD *)(v11 + 32) != *a3 || *(_QWORD *)(v11 + 40) >= a3[1]) )
      return sub_34A55B0((__int64)a1, a3);
    return 0;
  }
  v4 = *a3;
  v5 = a2[4];
  if ( *a3 < v5 || *a3 == v5 && a3[1] < a2[5] )
  {
    if ( (_QWORD *)a1[3] == a2 )
      return a2;
    v6 = sub_220EF80((__int64)a2);
    v7 = v6;
    if ( v4 > *(_QWORD *)(v6 + 32) || v4 == *(_QWORD *)(v6 + 32) && *(_QWORD *)(v6 + 40) < a3[1] )
    {
      result = 0;
      if ( *(_QWORD *)(v7 + 24) )
        return a2;
      return result;
    }
    return sub_34A55B0((__int64)a1, a3);
  }
  if ( v4 <= v5 && a2[5] >= a3[1] )
    return a2;
  if ( (_QWORD *)a1[4] == a2 )
    return 0;
  v9 = sub_220EEE0((__int64)a2);
  v10 = v9;
  if ( v4 >= *(_QWORD *)(v9 + 32) && (v4 != *(_QWORD *)(v9 + 32) || a3[1] >= *(_QWORD *)(v9 + 40)) )
    return sub_34A55B0((__int64)a1, a3);
  result = 0;
  if ( a2[3] )
    return (_QWORD *)v10;
  return result;
}
