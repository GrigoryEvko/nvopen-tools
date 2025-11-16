// Function: sub_38D4B70
// Address: 0x38d4b70
//
unsigned __int64 __fastcall sub_38D4B70(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned __int64 result; // rax

  (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)a1 + 512LL))(a1, a2, 0, 1, a3);
  result = sub_38D4B30(a1);
  *(_BYTE *)(result + 52) |= 1u;
  return result;
}
