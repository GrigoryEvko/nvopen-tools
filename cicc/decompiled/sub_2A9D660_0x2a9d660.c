// Function: sub_2A9D660
// Address: 0x2a9d660
//
_BOOL8 __fastcall sub_2A9D660(__int64 a1)
{
  bool v1; // r8
  _BOOL8 result; // rax

  v1 = sub_B46500((unsigned __int8 *)a1);
  result = 0;
  if ( !v1 )
    return !(*(_WORD *)(a1 + 2) & 1);
  return result;
}
