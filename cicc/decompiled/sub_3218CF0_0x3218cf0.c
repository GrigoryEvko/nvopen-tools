// Function: sub_3218CF0
// Address: 0x3218cf0
//
__int64 __fastcall sub_3218CF0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD **v4; // rdi
  void (*v5)(); // rax

  v4 = *(_QWORD ***)(a1 + 8);
  v5 = *(void (**)())(*v4[28] + 120LL);
  if ( v5 != nullsub_98 )
  {
    ((void (__fastcall *)(_QWORD *, __int64, __int64))v5)(v4[28], a3, 1);
    v4 = *(_QWORD ***)(a1 + 8);
  }
  return ((__int64 (__fastcall *)(_QWORD **, __int64, _QWORD))(*v4)[52])(v4, a2, 0);
}
