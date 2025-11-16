// Function: sub_2257420
// Address: 0x2257420
//
__int64 __fastcall sub_2257420(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 result; // rax

  result = sub_22578F0(a2);
  if ( !(_BYTE)result )
    return sub_2252230(a1, a2, a3);
  *a3 += 32;
  return result;
}
