// Function: sub_2243070
// Address: 0x2243070
//
__int64 __fastcall sub_2243070(__int64 a1, __int64 a2, __int64 a3)
{
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 8) = a3 != 0;
  *(_QWORD *)a1 = off_4A07AC0;
  *(_QWORD *)(a1 + 32) = sub_2208EB0();
  return sub_22567C0(a1, 0);
}
