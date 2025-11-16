// Function: sub_2D2B810
// Address: 0x2d2b810
//
_QWORD *__fastcall sub_2D2B810(_QWORD *a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  _QWORD *v5; // r8
  _QWORD *v6; // rax
  unsigned __int64 v7; // rcx

  v5 = *(_QWORD **)(*a1 + 8 * a2);
  if ( v5 )
  {
    v6 = (_QWORD *)*v5;
    v7 = *(_QWORD *)(*v5 + 64LL);
    while ( a4 != v7 || *a3 != v6[1] )
    {
      if ( !*v6 )
        return 0;
      v7 = *(_QWORD *)(*v6 + 64LL);
      v5 = v6;
      if ( a2 != v7 % a1[1] )
        return 0;
      v6 = (_QWORD *)*v6;
    }
  }
  return v5;
}
