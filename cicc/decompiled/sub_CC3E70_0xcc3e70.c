// Function: sub_CC3E70
// Address: 0xcc3e70
//
__int64 __fastcall sub_CC3E70(unsigned __int8 a1)
{
  if ( a1 <= 3u )
    return (unsigned int)(1 << a1);
  if ( (unsigned __int8)(a1 - 5) > 2u )
    BUG();
  return (unsigned int)(1 << (8 - a1)) | 0x100000000LL;
}
