// Function: sub_DCFC20
// Address: 0xdcfc20
//
__int64 __fastcall sub_DCFC20(_QWORD **a1, __int64 a2)
{
  _QWORD *v2; // r13
  _QWORD *v3; // r8
  __int64 result; // rax

  v2 = (_QWORD *)**a1;
  v3 = sub_DCFA50(a1[2], *a1[1], a2);
  result = 0;
  if ( v2 == v3 )
  {
    *a1[3] = *a1[1];
    *a1[4] = a2;
    return 1;
  }
  return result;
}
