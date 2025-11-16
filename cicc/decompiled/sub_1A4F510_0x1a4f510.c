// Function: sub_1A4F510
// Address: 0x1a4f510
//
__int64 __fastcall sub_1A4F510(__int64 a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi

  v2 = 24;
  if ( a2 != -2 )
    v2 = 24LL * (unsigned int)(2 * a2 + 3);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v3 = *(_QWORD *)(a1 - 8);
  else
    v3 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  return *(_QWORD *)(v3 + v2);
}
