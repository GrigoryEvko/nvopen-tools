// Function: sub_E99280
// Address: 0xe99280
//
void __fastcall sub_E99280(_QWORD **a1, __int64 a2, unsigned __int8 a3)
{
  void (*v4)(); // r13
  unsigned __int64 v5; // rax

  if ( a2 )
  {
    v4 = (void (*)())(*a1)[73];
    v5 = sub_E81A90(a2, a1[1], 0, 0);
    if ( v4 != nullsub_361 )
      ((void (__fastcall *)(_QWORD **, unsigned __int64, _QWORD, _QWORD))v4)(a1, v5, a3, 0);
  }
}
