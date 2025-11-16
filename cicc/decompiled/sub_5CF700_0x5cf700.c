// Function: sub_5CF700
// Address: 0x5cf700
//
void __fastcall sub_5CF700(__int64 *a1)
{
  while ( a1 )
  {
    *((_BYTE *)a1 + 11) |= 1u;
    a1 = (__int64 *)*a1;
  }
}
