// Function: sub_E3F7E0
// Address: 0xe3f7e0
//
char __fastcall sub_E3F7E0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  if ( a2 == 15 )
  {
    if ( *(_QWORD *)a1 == 0x7470656378657066LL
      && *(_DWORD *)(a1 + 8) == 1852270894
      && *(_WORD *)(a1 + 12) == 29295
      && *(_BYTE *)(a1 + 14) == 101 )
    {
      return 0;
    }
    if ( *(_QWORD *)a1 == 0x7470656378657066LL
      && *(_DWORD *)(a1 + 8) == 1920234286
      && *(_WORD *)(a1 + 12) == 25449
      && *(_BYTE *)(a1 + 14) == 116 )
    {
      return 2;
    }
    return a4;
  }
  if ( a2 != 16 || *(_QWORD *)a1 ^ 0x7470656378657066LL | *(_QWORD *)(a1 + 8) ^ 0x7061727479616D2ELL )
    return a4;
  return 1;
}
