// Function: sub_11D2BF0
// Address: 0x11d2bf0
//
__int64 __fastcall sub_11D2BF0(__int64 a1, __int64 a2)
{
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  *(_QWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 48) = a2;
  return a1 + 32;
}
