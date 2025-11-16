// Function: sub_14E96A0
// Address: 0x14e96a0
//
__int64 __fastcall sub_14E96A0(__int64 a1, __int64 a2)
{
  int v3; // eax

  if ( a2 == 17 )
  {
    if ( *(_QWORD *)a1 ^ 0x4E4943414D5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6E696665645F4F46LL
      || (v3 = 0, *(_BYTE *)(a1 + 16) != 101) )
    {
      v3 = 1;
    }
    if ( !v3 )
      return 1;
    return 0xFFFFFFFFLL;
  }
  if ( a2 == 16 )
    return (*(_QWORD *)a1 ^ 0x4E4943414D5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6665646E755F4F46LL) == 0 ? 2 : -1;
  if ( a2 != 21 )
  {
    if ( a2 == 19
      && !(*(_QWORD *)a1 ^ 0x4E4943414D5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x665F646E655F4F46LL)
      && *(_WORD *)(a1 + 16) == 27753
      && *(_BYTE *)(a1 + 18) == 101 )
    {
      return 4;
    }
    return 0xFFFFFFFFLL;
  }
  if ( *(_QWORD *)a1 ^ 0x4E4943414D5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x74726174735F4F46LL
    || *(_DWORD *)(a1 + 16) != 1818846815
    || *(_BYTE *)(a1 + 20) != 101 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x4E4943414D5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6F646E65765F4F46LL)
      && *(_DWORD *)(a1 + 16) == 2019909490
      && *(_BYTE *)(a1 + 20) == 116 )
    {
      return 255;
    }
    return 0xFFFFFFFFLL;
  }
  return 3;
}
