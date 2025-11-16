// Function: sub_12E9850
// Address: 0x12e9850
//
__int64 __fastcall sub_12E9850(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 (__fastcall *v6)(__int64, __int64, _QWORD); // rbx
  __int64 v7; // rax

  v6 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL);
  v7 = sub_1841180(a1, a2, a3, a4, a5, a6);
  return v6(a2, v7, 0);
}
