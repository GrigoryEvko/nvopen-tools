// Function: sub_8CC270
// Address: 0x8cc270
//
_QWORD *__fastcall sub_8CC270(__int64 *a1)
{
  __int64 v1; // r13
  _QWORD *result; // rax
  __int64 v3; // r13
  __int64 v4; // rax

  v1 = *a1;
  result = (_QWORD *)sub_8C6B40(*a1);
  if ( !(_DWORD)result )
  {
    if ( a1[4] )
      return result;
    return sub_8C7090(11, (__int64)a1);
  }
  result = sub_8C8170(v1, *(_QWORD *)(*(_QWORD *)v1 + 32LL));
  if ( result || (v4 = sub_8C6F20(v1), (result = sub_8C8170(v1, v4)) != 0) )
  {
    v3 = result[11];
    sub_8CC0D0((__int64)a1, v3);
    result = &dword_4F077C4;
    if ( dword_4F077C4 != 2 )
      return (_QWORD *)sub_8DED30(a1[19], *(_QWORD *)(v3 + 152), 261);
  }
  else if ( !a1[4] )
  {
    return sub_8C7090(11, (__int64)a1);
  }
  return result;
}
