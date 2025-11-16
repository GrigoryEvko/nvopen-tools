// Function: sub_1D5B320
// Address: 0x1d5b320
//
__int64 __fastcall sub_1D5B320(__int64 a1)
{
  unsigned int v1; // r8d

  sub_16348C0(a1);
  v1 = 0;
  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) == 2 )
    LOBYTE(v1) = *(_BYTE *)(*(_QWORD *)(a1 - 24) + 16LL) == 13;
  return v1;
}
