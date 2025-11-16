// Function: sub_222AA00
// Address: 0x222aa00
//
const char *__fastcall sub_222AA00(__int64 a1, __int64 a2)
{
  const char *result; // rax

  *(_DWORD *)(a1 + 8) = a2 != 0;
  *(_QWORD *)a1 = off_4A061E0;
  *(_QWORD *)(a1 + 16) = sub_2208E60(a1, a2);
  result = sub_2208EB0();
  *(_QWORD *)(a1 + 24) = result;
  return result;
}
