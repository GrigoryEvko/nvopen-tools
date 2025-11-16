// Function: sub_A77BC0
// Address: 0xa77bc0
//
__int64 __fastcall sub_A77BC0(__int64 **a1, unsigned __int16 a2)
{
  if ( HIBYTE(a2) )
    return sub_A77B60(a1, 94, 1LL << a2);
  else
    return (__int64)a1;
}
