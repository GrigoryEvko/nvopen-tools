// Function: sub_139BE10
// Address: 0x139be10
//
__int64 __fastcall sub_139BE10(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( *(_BYTE *)(sub_1648700(a2) + 16) != 25 || (result = *(unsigned __int8 *)(a1 + 8), (_BYTE)result) )
  {
    *(_BYTE *)(a1 + 9) = 1;
    return 1;
  }
  return result;
}
