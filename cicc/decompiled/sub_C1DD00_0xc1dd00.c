// Function: sub_C1DD00
// Address: 0xc1dd00
//
_QWORD *__fastcall sub_C1DD00(_QWORD *a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  _QWORD *v5; // r8
  _QWORD *v6; // rax
  unsigned __int64 v7; // rcx

  v5 = *(_QWORD **)(*a1 + 8 * a2);
  if ( v5 )
  {
    v6 = (_QWORD *)*v5;
    v7 = *(_QWORD *)(*v5 + 192LL);
    while ( v7 != a4 || v6[1] != *a3 )
    {
      if ( !*v6 )
        return 0;
      v7 = *(_QWORD *)(*v6 + 192LL);
      v5 = v6;
      if ( a2 != v7 % a1[1] )
        return 0;
      v6 = (_QWORD *)*v6;
    }
  }
  return v5;
}
