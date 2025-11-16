// Function: sub_14ECA90
// Address: 0x14eca90
//
__int64 __fastcall sub_14ECA90(__int64 *a1)
{
  __int64 result; // rax

  result = *a1;
  if ( (*a1 & 1) != 0 || (result & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(a1);
  return result;
}
