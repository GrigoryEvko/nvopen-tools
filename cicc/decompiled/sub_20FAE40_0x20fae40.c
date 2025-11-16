// Function: sub_20FAE40
// Address: 0x20fae40
//
_QWORD *__fastcall sub_20FAE40(_QWORD *a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  _QWORD *v5; // r10
  _QWORD *v6; // rax
  unsigned __int64 v7; // rcx
  _QWORD *v8; // r8

  v5 = *(_QWORD **)(*a1 + 8 * a2);
  if ( !v5 )
    return 0;
  v6 = (_QWORD *)*v5;
  v7 = *(_QWORD *)(*v5 + 208LL);
  while ( v7 != a4 || *a3 != v6[1] || a3[1] != v6[2] )
  {
    v8 = (_QWORD *)*v6;
    if ( !*v6 )
      return v8;
    v7 = v8[26];
    v5 = v6;
    if ( a2 != v7 % a1[1] )
      return 0;
    v6 = (_QWORD *)*v6;
  }
  return v5;
}
