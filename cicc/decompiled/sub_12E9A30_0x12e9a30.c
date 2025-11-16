// Function: sub_12E9A30
// Address: 0x12e9a30
//
__int64 __fastcall sub_12E9A30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 (__fastcall *v3)(__int64, __int64, _QWORD); // rbx
  __int64 v4; // rax

  v3 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL);
  v4 = sub_1CC60B0(a1, a2, a3);
  return v3(a2, v4, 0);
}
