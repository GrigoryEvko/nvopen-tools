// Function: sub_396E900
// Address: 0x396e900
//
__int64 __fastcall sub_396E900(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 (__fastcall *v4)(__int64, __int64, __int64, _QWORD); // rbx
  __int64 v5; // rax

  v4 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)a2 + 1016LL);
  v5 = sub_396E580(a1);
  return v4(a2, a3, v5, 0);
}
