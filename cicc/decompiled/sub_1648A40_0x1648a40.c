// Function: sub_1648A40
// Address: 0x1648a40
//
__int64 __fastcall sub_1648A40(__int64 a1)
{
  __int64 v1; // rdi

  v1 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  return v1 - *(_QWORD *)(v1 - 8) - 8;
}
