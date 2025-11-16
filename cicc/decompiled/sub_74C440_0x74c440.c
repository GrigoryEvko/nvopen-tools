// Function: sub_74C440
// Address: 0x74c440
//
void __fastcall sub_74C440(int a1, __int64 a2, __int64 a3)
{
  void (__fastcall *v3)(__int64, _QWORD); // rax

  if ( a1 )
  {
    v3 = *(void (__fastcall **)(__int64, _QWORD))(a3 + 40);
    if ( v3 )
      v3(a2, 0);
    else
      sub_74C3E0(a2, (__int64 (__fastcall **)(char *, _QWORD))a3);
  }
  else if ( a2 )
  {
    sub_74C380(a2, (__int64 (__fastcall **)(char *, _QWORD))a3);
  }
}
