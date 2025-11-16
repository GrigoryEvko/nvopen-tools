// Function: sub_C7D670
// Address: 0xc7d670
//
__int64 __fastcall sub_C7D670(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_22077E0(a1, a2, &unk_435FF63);
  if ( !result )
    sub_C64F00("Buffer allocation failed", 1u);
  return result;
}
