// Function: sub_12E9E80
// Address: 0x12e9e80
//
__int64 __fastcall sub_12E9E80(__int64 a1, __int64 a2)
{
  __int64 (__fastcall *v2)(__int64, __int64, _QWORD); // rbx
  __int64 v3; // rax

  v2 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL);
  v3 = sub_184CD60(a1, a2);
  return v2(a2, v3, 0);
}
