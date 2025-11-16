// Function: sub_277ABA0
// Address: 0x277aba0
//
__int64 __fastcall sub_277ABA0(__int64 a1)
{
  __int64 result; // rax

  result = a1;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    return *(_QWORD *)(a1 - 8) + 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  return result;
}
