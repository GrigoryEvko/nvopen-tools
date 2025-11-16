// Function: sub_24F2690
// Address: 0x24f2690
//
void __fastcall sub_24F2690(_QWORD *a1)
{
  void (__fastcall *v1)(_QWORD *, _QWORD *, __int64); // rax

  *a1 = &unk_4A32DB0;
  v1 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[5];
  if ( v1 )
    v1(a1 + 3, a1 + 3, 3);
  j_j___libc_free_0((unsigned __int64)a1);
}
