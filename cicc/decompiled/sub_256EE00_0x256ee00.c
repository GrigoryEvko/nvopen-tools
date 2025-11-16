// Function: sub_256EE00
// Address: 0x256ee00
//
__int64 __fastcall sub_256EE00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // eax
  __int64 v9; // rdi
  __int64 v10; // rsi

  v6 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( !v6 )
    BUG();
  if ( (unsigned int)**(unsigned __int8 **)(a2 - 32LL * v6) - 12 <= 1 )
    return 1;
  v9 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v10 = *(_QWORD *)(a2 - 8);
  else
    v10 = a2 - 32LL * v6;
  **(_BYTE **)a1 |= sub_256E5A0(v9, v10, *(unsigned __int8 **)(a1 + 16), a4, a5, a6);
  return 1;
}
