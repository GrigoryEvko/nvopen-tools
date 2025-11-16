// Function: sub_6E6840
// Address: 0x6e6840
//
__int64 __fastcall sub_6E6840(__int64 a1)
{
  __int64 result; // rax

  sub_6E6450(a1);
  sub_6E2DD0(a1, 0);
  result = sub_72C930(a1);
  *(_DWORD *)(a1 + 17) &= 0xFFE9C500;
  *(_QWORD *)a1 = result;
  return result;
}
