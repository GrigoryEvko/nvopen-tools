// Function: sub_9C66B0
// Address: 0x9c66b0
//
__int64 __fastcall sub_9C66B0(__int64 *a1)
{
  __int64 result; // rax

  result = *a1;
  if ( (*a1 & 1) != 0 || (result & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(a1);
  return result;
}
