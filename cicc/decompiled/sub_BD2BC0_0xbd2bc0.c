// Function: sub_BD2BC0
// Address: 0xbd2bc0
//
__int64 __fastcall sub_BD2BC0(__int64 a1)
{
  __int64 v1; // rdi

  v1 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  return v1 - *(_QWORD *)(v1 - 8) - 8;
}
