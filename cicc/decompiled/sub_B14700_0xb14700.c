// Function: sub_B14700
// Address: 0xb14700
//
__int64 __fastcall sub_B14700(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax

  v2 = (*(__int64 (__fastcall **)(__int64, const char *))(*(_QWORD *)a2 + 48LL))(
         a2,
         "ignoring debug info with an invalid version (");
  v3 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v2 + 104LL))(v2, *(unsigned int *)(a1 + 24));
  v4 = (*(__int64 (__fastcall **)(__int64, const char *))(*(_QWORD *)v3 + 48LL))(v3, ") in ");
  return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 144LL))(v4, *(_QWORD *)(a1 + 16));
}
