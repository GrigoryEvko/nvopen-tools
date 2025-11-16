// Function: sub_1BBF440
// Address: 0x1bbf440
//
__int64 __fastcall sub_1BBF440(__int64 a1)
{
  if ( *(char *)(a1 + 23) >= 0 )
    BUG();
  return *(unsigned int *)(sub_1648A40(a1) + 8);
}
