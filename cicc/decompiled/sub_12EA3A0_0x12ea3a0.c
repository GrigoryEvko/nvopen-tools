// Function: sub_12EA3A0
// Address: 0x12ea3a0
//
__int64 __fastcall sub_12EA3A0(_DWORD *a1, __int64 a2)
{
  __int64 (__fastcall *v2)(__int64, __int64, _QWORD); // rbx
  __int64 v3; // rax

  v2 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL);
  v3 = sub_19B73C0(a1[4], a1[5], a1[6], a1[7], a1[8], a1[9], a1[10]);
  return v2(a2, v3, 0);
}
