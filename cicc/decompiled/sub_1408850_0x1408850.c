// Function: sub_1408850
// Address: 0x1408850
//
__int64 __fastcall sub_1408850(unsigned __int64 a1)
{
  int v1; // ecx
  unsigned __int64 v2; // rdx

  v1 = a1 & 7;
  if ( v1 == 1 )
    return a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v1 == 2 )
    return a1 & 0xFFFFFFFFFFFFFFF8LL | 2;
  v2 = a1 & 0xFFFFFFFFFFFFFFF8LL | 6;
  if ( v1 == 3 && (v2 = 6, a1 >> 61 == 2) )
    return 4;
  else
    return v2;
}
