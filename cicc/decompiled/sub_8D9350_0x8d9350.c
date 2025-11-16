// Function: sub_8D9350
// Address: 0x8d9350
//
__int64 __fastcall sub_8D9350(__int64 a1, _QWORD *a2)
{
  int v2; // r13d
  __int16 v3; // r12
  __int64 result; // rax

  v2 = dword_4F07508[0];
  v3 = dword_4F07508[1];
  *(_QWORD *)dword_4F07508 = *a2;
  result = sub_8D8C50(a1, (__int64 (__fastcall *)(__int64, unsigned int *))sub_8DD400, (__int64)sub_8D1280, 0x64Bu);
  dword_4F07508[0] = v2;
  LOWORD(dword_4F07508[1]) = v3;
  return result;
}
