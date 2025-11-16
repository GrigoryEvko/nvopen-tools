// Function: sub_12EA120
// Address: 0x12ea120
//
__int64 __fastcall sub_12EA120(__int64 a1, __int64 a2)
{
  __int64 (__fastcall *v2)(__int64, __int64, _QWORD); // rbx
  __int64 v3; // rax

  v2 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL);
  v3 = sub_1B7FDF0(*(unsigned int *)(a1 + 16));
  return v2(a2, v3, 0);
}
