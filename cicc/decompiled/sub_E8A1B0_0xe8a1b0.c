// Function: sub_E8A1B0
// Address: 0xe8a1b0
//
__int64 __fastcall sub_E8A1B0(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 result; // rax

  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD, __int64, _QWORD))(*a1 + 608LL))(a1, a2, 0, 1, a4);
  result = a1[36];
  *(_BYTE *)(result + 31) |= 1u;
  *(_QWORD *)(result + 48) = a3;
  return result;
}
