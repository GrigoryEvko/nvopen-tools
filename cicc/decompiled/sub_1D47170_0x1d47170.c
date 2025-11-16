// Function: sub_1D47170
// Address: 0x1d47170
//
__int64 __fastcall sub_1D47170(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = &unk_49F9A08;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[5];
  if ( v1 )
    v1(a1 + 3, a1 + 3, 3);
  *(_QWORD *)(a1[2] + 664LL) = a1[1];
  return j_j___libc_free_0(a1, 56);
}
