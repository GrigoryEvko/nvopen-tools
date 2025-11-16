// Function: sub_34B6050
// Address: 0x34b6050
//
_QWORD *__fastcall sub_34B6050(_QWORD *a1, __int64 a2, unsigned __int64 *a3)
{
  unsigned __int64 v4; // r14
  __int64 v5; // r8
  __int64 v6; // rax
  __int64 v8; // rax

  if ( (_QWORD *)a2 == a1 + 1 )
  {
    if ( !a1[5] || *a3 <= *(_QWORD *)(a1[4] + 32LL) )
      return sub_34B5F00((__int64)a1, a3);
  }
  else
  {
    v4 = *a3;
    if ( *(_QWORD *)(a2 + 32) > *a3 )
    {
      v5 = a1[3];
      if ( v5 == a2 )
        return (_QWORD *)v5;
      v6 = sub_220EF80(a2);
      if ( *(_QWORD *)(v6 + 32) < v4 )
      {
        v5 = 0;
        if ( *(_QWORD *)(v6 + 24) )
          return (_QWORD *)a2;
        return (_QWORD *)v5;
      }
      return sub_34B5F00((__int64)a1, a3);
    }
    if ( *(_QWORD *)(a2 + 32) >= *a3 )
      return (_QWORD *)a2;
    if ( a1[4] != a2 )
    {
      v8 = sub_220EEE0(a2);
      if ( *(_QWORD *)(v8 + 32) > v4 )
      {
        v5 = 0;
        if ( *(_QWORD *)(a2 + 24) )
          return (_QWORD *)v8;
        return (_QWORD *)v5;
      }
      return sub_34B5F00((__int64)a1, a3);
    }
  }
  return 0;
}
