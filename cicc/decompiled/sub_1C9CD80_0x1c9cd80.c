// Function: sub_1C9CD80
// Address: 0x1c9cd80
//
_QWORD *__fastcall sub_1C9CD80(_QWORD *a1, _QWORD *a2, unsigned __int64 *a3)
{
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rax
  bool v6; // dl
  _QWORD *result; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rdx
  bool v11; // al
  bool v12; // dl
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  unsigned __int64 v16; // rdx
  bool v17; // al

  if ( a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_1C9CCC0((__int64)a1, a3);
    v15 = a1[4];
    v16 = *(_QWORD *)(v15 + 32);
    v17 = *a3 > v16;
    if ( *a3 == v16 )
      v17 = a3[1] > *(_QWORD *)(v15 + 40);
    if ( !v17 )
      return sub_1C9CCC0((__int64)a1, a3);
    return 0;
  }
  v4 = *a3;
  v5 = a2[4];
  v6 = *a3 < v5;
  if ( v5 == v4 )
    v6 = a2[5] > a3[1];
  if ( v6 )
  {
    result = a2;
    if ( (_QWORD *)a1[3] == a2 )
      return result;
    v8 = sub_220EF80(a2);
    v9 = *(_QWORD *)(v8 + 32);
    v10 = v8;
    v11 = v9 < v4;
    if ( v4 == v9 )
      v11 = *(_QWORD *)(v10 + 40) < a3[1];
    if ( v11 )
    {
      result = 0;
      if ( *(_QWORD *)(v10 + 24) )
        return a2;
      return result;
    }
    return sub_1C9CCC0((__int64)a1, a3);
  }
  v12 = v5 < v4;
  if ( v5 == v4 )
    v12 = a2[5] < a3[1];
  if ( !v12 )
    return a2;
  if ( (_QWORD *)a1[4] == a2 )
    return 0;
  v13 = sub_220EEE0(a2);
  v14 = v13;
  if ( v4 == *(_QWORD *)(v13 + 32) )
  {
    if ( a3[1] >= *(_QWORD *)(v13 + 40) )
      return sub_1C9CCC0((__int64)a1, a3);
  }
  else if ( v4 >= *(_QWORD *)(v13 + 32) )
  {
    return sub_1C9CCC0((__int64)a1, a3);
  }
  result = 0;
  if ( a2[3] )
    return (_QWORD *)v14;
  return result;
}
