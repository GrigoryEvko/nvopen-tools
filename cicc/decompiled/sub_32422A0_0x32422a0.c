// Function: sub_32422A0
// Address: 0x32422a0
//
void __fastcall sub_32422A0(_QWORD *a1, unsigned int a2, unsigned int a3)
{
  void (__fastcall *v4)(_QWORD *, __int64, _QWORD); // rax

  if ( a2 )
  {
    v4 = *(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a1 + 8LL);
    if ( a3 | a2 & 7 )
    {
      v4(a1, 157, 0);
      (*(void (__fastcall **)(_QWORD *, _QWORD))(*a1 + 24LL))(a1, a2);
      (*(void (__fastcall **)(_QWORD *, _QWORD))(*a1 + 24LL))(a1, a3);
    }
    else
    {
      v4(a1, 147, 0);
      (*(void (__fastcall **)(_QWORD *, _QWORD))(*a1 + 24LL))(a1, a2 >> 3);
    }
    a1[11] += a2;
  }
}
