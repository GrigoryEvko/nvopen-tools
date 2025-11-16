// Function: sub_2B0D930
// Address: 0x2b0d930
//
unsigned __int64 __fastcall sub_2B0D930(unsigned __int64 a1, unsigned int a2)
{
  if ( (a1 & 1) != 0 )
    return (((a1 >> 1) & ~(-1LL << (a1 >> 58))) >> a2) & 1;
  else
    return (*(_QWORD *)(*(_QWORD *)a1 + 8LL * (a2 >> 6)) >> a2) & 1LL;
}
