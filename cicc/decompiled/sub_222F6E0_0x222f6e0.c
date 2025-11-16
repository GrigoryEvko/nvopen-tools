// Function: sub_222F6E0
// Address: 0x222f6e0
//
__int64 __fastcall sub_222F6E0(__int64 a1, __int64 a2, __int64 a3)
{
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 8) = a3 != 0;
  *(_QWORD *)a1 = off_4A06CB8;
  *(_QWORD *)(a1 + 32) = sub_2208EB0();
  return sub_2255E40(a1, 0);
}
