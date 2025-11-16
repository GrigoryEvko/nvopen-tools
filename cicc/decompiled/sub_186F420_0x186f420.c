// Function: sub_186F420
// Address: 0x186f420
//
__int64 __fastcall sub_186F420(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = off_49F1908;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[22];
  if ( v1 )
    v1(a1 + 20, a1 + 20, 3);
  sub_1636790(a1);
  return j_j___libc_free_0(a1, 192);
}
