// Function: sub_DFE910
// Address: 0xdfe910
//
_QWORD *__fastcall sub_DFE910(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r13
  __int64 (__fastcall *v4)(_QWORD *); // rax

  v2 = *a2;
  *a2 = 0;
  v3 = (_QWORD *)*a1;
  *a1 = v2;
  if ( !v3 )
    return a1;
  v4 = *(__int64 (__fastcall **)(_QWORD *))(*v3 + 8LL);
  if ( v4 == sub_DFE780 )
  {
    *v3 = off_4979D10;
    nullsub_197();
    j_j___libc_free_0(v3, 16);
    return a1;
  }
  v4(v3);
  return a1;
}
