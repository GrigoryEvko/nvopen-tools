// Function: sub_7A7520
// Address: 0x7a7520
//
__int64 __fastcall sub_7A7520(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  result = 0;
  if ( unk_4F0698C <= a1 && unk_4F06988 >= a1 && (a1 & (a1 - 1)) == 0 )
  {
    *a2 = a1;
    return 1;
  }
  return result;
}
