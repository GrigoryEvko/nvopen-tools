// Function: sub_D04110
// Address: 0xd04110
//
__int64 __fastcall sub_D04110(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v7; // rsi

  if ( a2 != a3 )
    return 0;
  if ( !*(_BYTE *)(a4 + 512) || *(_BYTE *)a2 <= 0x1Cu || sub_AA5B70(*(_QWORD *)(a2 + 40)) )
    return 1;
  v7 = 0;
  if ( *(_BYTE *)(a4 + 513) )
    v7 = *(_BYTE **)(a1 + 32);
  return sub_D00920(a2, v7, 0);
}
