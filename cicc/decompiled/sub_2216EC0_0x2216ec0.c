// Function: sub_2216EC0
// Address: 0x2216ec0
//
__int64 __fastcall sub_2216EC0(__int64 a1, __int64 a2)
{
  *(_DWORD *)(a1 + 8) = a2 != 0;
  *(_QWORD *)a1 = off_4A05450;
  *(_QWORD *)(a1 + 16) = sub_2208E60();
  *(_BYTE *)(a1 + 24) = 0;
  return sub_2217600(a1);
}
