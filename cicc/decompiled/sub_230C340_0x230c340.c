// Function: sub_230C340
// Address: 0x230c340
//
void __fastcall sub_230C340(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = &unk_4A15678;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[3];
  if ( v1 )
    v1(a1 + 1, a1 + 1, 3);
  j_j___libc_free_0((unsigned __int64)a1);
}
