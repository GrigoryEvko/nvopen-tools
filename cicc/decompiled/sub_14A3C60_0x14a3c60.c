// Function: sub_14A3C60
// Address: 0x14a3c60
//
_QWORD *__fastcall sub_14A3C60(_QWORD *a1, __int64 *a2)
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
  if ( v4 == sub_14A3AF0 )
  {
    *v3 = off_4984830;
    nullsub_542();
    j_j___libc_free_0(v3, 16);
    return a1;
  }
  v4(v3);
  return a1;
}
