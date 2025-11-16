// Function: sub_BD2910
// Address: 0xbd2910
//
__int64 __fastcall sub_BD2910(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdi

  v1 = *(_QWORD *)(a1 + 24);
  if ( (*(_BYTE *)(v1 + 7) & 0x40) != 0 )
    v2 = a1 - *(_QWORD *)(v1 - 8);
  else
    v2 = a1 - (v1 - 32LL * (*(_DWORD *)(v1 + 4) & 0x7FFFFFF));
  return v2 >> 5;
}
