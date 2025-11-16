// Function: sub_33C8920
// Address: 0x33c8920
//
void __fastcall sub_33C8920(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = &unk_4A36708;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[5];
  if ( v1 )
    v1(a1 + 3, a1 + 3, 3);
  *(_QWORD *)(a1[2] + 768LL) = a1[1];
  j_j___libc_free_0((unsigned __int64)a1);
}
