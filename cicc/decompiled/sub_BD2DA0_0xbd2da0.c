// Function: sub_BD2DA0
// Address: 0xbd2da0
//
__int64 __fastcall sub_BD2DA0(__int64 a1)
{
  __int64 v1; // rax
  int v2; // edx
  __int64 result; // rax

  v1 = sub_22077B0(a1 + 8);
  v2 = *(_DWORD *)(v1 + 12);
  *(_QWORD *)v1 = 0;
  result = v1 + 8;
  *(_DWORD *)(result + 4) = v2 & 0x38000000 | 0x40000000;
  return result;
}
