// Function: sub_1455F60
// Address: 0x1455f60
//
__int64 __fastcall sub_1455F60(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdi

  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v2 = *(_QWORD *)(a1 - 8);
  else
    v2 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  return *(_QWORD *)(v2 + 24LL * a2);
}
