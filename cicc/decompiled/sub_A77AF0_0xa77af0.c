// Function: sub_A77AF0
// Address: 0xa77af0
//
__int64 __fastcall sub_A77AF0(__int64 *a1, __int64 a2, unsigned int *a3)
{
  __int64 v4; // rsi
  __int64 v5; // rdx

  v4 = a2 << 32;
  v5 = 0xFFFFFFFFLL;
  if ( *((_BYTE *)a3 + 4) )
    v5 = *a3;
  return sub_A778C0(a1, 88, v4 | v5);
}
