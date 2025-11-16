// Function: sub_A77B90
// Address: 0xa77b90
//
__int64 __fastcall sub_A77B90(__int64 **a1, unsigned __int16 a2)
{
  if ( HIBYTE(a2) )
    return sub_A77B60(a1, 86, 1LL << a2);
  else
    return (__int64)a1;
}
