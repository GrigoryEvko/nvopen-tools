// Function: sub_3243EC0
// Address: 0x3243ec0
//
__int64 __fastcall sub_3243EC0(__int64 a1, unsigned int a2)
{
  void (__fastcall *v2)(__int64, __int64, _QWORD); // rax

  v2 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 8LL);
  if ( a2 > 0x22 )
  {
    v2(a1, 49, 0);
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 8LL))(a1, 16, 0);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, a2);
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 8LL))(a1, 36, 0);
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 8LL))(a1, 49, 0);
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 8LL))(a1, 28, 0);
  }
  else
  {
    v2(a1, 16, 0);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 24LL))(a1, (1LL << a2) - 1);
  }
  return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 8LL))(a1, 26, 0);
}
