// Function: sub_2BEE9E0
// Address: 0x2bee9e0
//
__int64 __fastcall sub_2BEE9E0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v3; // rdi

  v3 = *(_QWORD *)(a2 + 120);
  LOBYTE(v2) = *(_DWORD *)(v3 + 16) != 0;
  sub_318DFC0(v3 + 8);
  return v2;
}
