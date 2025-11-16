// Function: sub_1683CD0
// Address: 0x1683cd0
//
void sub_1683CD0()
{
  unsigned __int64 i; // r12
  __int64 v1; // rdi

  for ( i = qword_4F9F320; qword_4F9F320; i = qword_4F9F320 )
  {
    v1 = *(_QWORD *)(i + 8);
    qword_4F9F320 = *(_QWORD *)(i + 16);
    (*(void (__fastcall **)(__int64))i)(v1);
    _libc_free(i);
  }
}
