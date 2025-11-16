// Function: sub_1ABC8A0
// Address: 0x1abc8a0
//
_QWORD *__fastcall sub_1ABC8A0(_QWORD *a1, _QWORD *a2, unsigned __int64 *a3)
{
  unsigned __int64 v4; // r14
  _QWORD *v5; // r8
  __int64 v6; // rax
  __int64 v8; // rax

  if ( a2 == a1 + 1 )
  {
    if ( !a1[5] || *a3 <= *(_QWORD *)(a1[4] + 32LL) )
      return sub_1ABC800((__int64)a1, a3);
  }
  else
  {
    v4 = *a3;
    if ( a2[4] > *a3 )
    {
      v5 = (_QWORD *)a1[3];
      if ( v5 == a2 )
        return v5;
      v6 = sub_220EF80(a2);
      if ( *(_QWORD *)(v6 + 32) < v4 )
      {
        v5 = 0;
        if ( *(_QWORD *)(v6 + 24) )
          return a2;
        return v5;
      }
      return sub_1ABC800((__int64)a1, a3);
    }
    if ( a2[4] >= *a3 )
      return a2;
    if ( (_QWORD *)a1[4] != a2 )
    {
      v8 = sub_220EEE0(a2);
      if ( *(_QWORD *)(v8 + 32) > v4 )
      {
        v5 = 0;
        if ( a2[3] )
          return (_QWORD *)v8;
        return v5;
      }
      return sub_1ABC800((__int64)a1, a3);
    }
  }
  return 0;
}
