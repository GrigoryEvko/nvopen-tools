// Function: sub_140CED0
// Address: 0x140ced0
//
__int64 __fastcall sub_140CED0(__int64 a1)
{
  *(_DWORD *)(a1 + 8) = 1;
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 24) = 1;
  *(_QWORD *)(a1 + 16) = 0;
  return a1;
}
