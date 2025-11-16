// Function: sub_DCEA30
// Address: 0xdcea30
//
_QWORD *__fastcall sub_DCEA30(_QWORD *a1, __int16 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rcx
  unsigned __int64 *v7; // rdx
  _QWORD *result; // rax

  v6 = *(unsigned int *)(a3 + 8);
  v7 = *(unsigned __int64 **)a3;
  if ( v6 == 1 )
    return (_QWORD *)*v7;
  result = sub_D96EA0((__int64)a1, a2, v7, v6, a5);
  if ( !result )
    return (_QWORD *)sub_DCE310(a1, a2, a3);
  return result;
}
