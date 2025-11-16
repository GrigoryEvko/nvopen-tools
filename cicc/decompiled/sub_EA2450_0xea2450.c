// Function: sub_EA2450
// Address: 0xea2450
//
__int64 __fastcall sub_EA2450(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 (__fastcall *v11)(__int64, __int64, __int64, __int64, __int64, __int64); // [rsp+8h] [rbp-38h]

  v11 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 48LL);
  v9 = sub_ECD6A0(a5);
  return v11(a1, a2, a3, a4, v9, a6);
}
