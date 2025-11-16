// Function: sub_399F630
// Address: 0x399f630
//
__int64 __fastcall sub_399F630(_DWORD *a1, __int64 a2)
{
  void (__fastcall **v2)(_DWORD *, __int64, _QWORD); // rax

  v2 = *(void (__fastcall ***)(_DWORD *, __int64, _QWORD))a1;
  a1[19] = 3;
  (*v2)(a1, 17, 0);
  return (*(__int64 (__fastcall **)(_DWORD *, __int64))(*(_QWORD *)a1 + 16LL))(a1, a2);
}
