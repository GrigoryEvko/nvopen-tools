// Function: sub_169E610
// Address: 0x169e610
//
__int64 __fastcall sub_169E610(__int64 a1, char *a2, __int64 a3, unsigned int a4)
{
  __int16 **v7; // rdi

  v7 = (__int16 **)(a1 + 8);
  if ( *(void **)(a1 + 8) == sub_16982C0() )
    return sub_169E440(v7, (__int64)a2, a3, a4);
  else
    return sub_169C210(v7, a2, a3, a4);
}
