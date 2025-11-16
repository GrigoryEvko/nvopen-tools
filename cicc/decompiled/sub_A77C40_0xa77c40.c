// Function: sub_A77C40
// Address: 0xa77c40
//
__int64 __fastcall sub_A77C40(__int64 **a1, __int64 a2, unsigned int *a3)
{
  __int64 v3; // rsi
  __int64 v4; // rax

  v3 = a2 << 32;
  v4 = 0xFFFFFFFFLL;
  if ( *((_BYTE *)a3 + 4) )
    v4 = *a3;
  return sub_A77C30(a1, v4 | v3);
}
