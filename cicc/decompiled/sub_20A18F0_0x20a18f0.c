// Function: sub_20A18F0
// Address: 0x20a18f0
//
bool __fastcall sub_20A18F0(__int64 a1)
{
  if ( *(_BYTE *)a1 )
    return (unsigned __int8)(*(_BYTE *)a1 - 2) <= 5u || (unsigned __int8)(*(_BYTE *)a1 - 14) <= 0x47u;
  else
    return sub_1F58CF0(a1);
}
