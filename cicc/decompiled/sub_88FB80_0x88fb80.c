// Function: sub_88FB80
// Address: 0x88fb80
//
__int64 __fastcall sub_88FB80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int128 *v7; // r13
  FILE *v8; // rsi

  if ( !a1 || !a2 )
    return 1;
  v7 = *(__int128 **)(a2 + 16);
  if ( (unsigned int)sub_739400(*(__int128 **)(a1 + 16), v7) )
    return 1;
  v8 = (FILE *)((char *)v7 + 8);
  if ( !v7 )
    v8 = (FILE *)(a3 + 8);
  sub_6854C0(0xBEBu, v8, a4);
  return 0;
}
