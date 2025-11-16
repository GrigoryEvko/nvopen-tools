// Function: sub_39ACB90
// Address: 0x39acb90
//
__int64 __fastcall sub_39ACB90(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  sub_39A9FA0((_QWORD *)a1, a2);
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)a1 = off_4A3FE58;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  v2 = sub_396DDB0(a2);
  result = 8 * (unsigned int)sub_15A9520(v2, 0);
  *(_BYTE *)(a1 + 27) = (_DWORD)result == 64;
  return result;
}
