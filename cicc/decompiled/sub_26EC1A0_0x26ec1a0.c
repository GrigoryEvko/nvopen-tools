// Function: sub_26EC1A0
// Address: 0x26ec1a0
//
_QWORD *__fastcall sub_26EC1A0(_QWORD *a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  _QWORD *v5; // r8
  _QWORD *v6; // rax
  unsigned __int64 v7; // rcx

  v5 = *(_QWORD **)(*a1 + 8 * a2);
  if ( v5 )
  {
    v6 = (_QWORD *)*v5;
    v7 = *(_QWORD *)(*v5 + 32LL);
    while ( v7 != a4 || *a3 != v6[1] || a3[1] != v6[2] )
    {
      if ( !*v6 )
        return 0;
      v7 = *(_QWORD *)(*v6 + 32LL);
      v5 = v6;
      if ( a2 != v7 % a1[1] )
        return 0;
      v6 = (_QWORD *)*v6;
    }
  }
  return v5;
}
