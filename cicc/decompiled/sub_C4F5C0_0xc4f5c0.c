// Function: sub_C4F5C0
// Address: 0xc4f5c0
//
__int64 __fastcall sub_C4F5C0(__int64 a1)
{
  unsigned __int16 v1; // ax

  v1 = *(unsigned __int8 *)(a1 + 13);
  LOBYTE(v1) = (unsigned __int8)v1 >> 1;
  return (v1 >> 3) & 1;
}
