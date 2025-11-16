// Function: sub_31B9400
// Address: 0x31b9400
//
_BOOL8 __fastcall sub_31B9400(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  int v5; // ecx
  _BOOL8 result; // rax

  v4 = sub_31B90D0(a2, a3);
  if ( v4 == 3 )
    return 0;
  v5 = v4;
  if ( v4 > 3 )
  {
    result = 1;
    if ( v5 == 4 )
      return result;
    if ( v5 == 5 )
      return 0;
LABEL_8:
    BUG();
  }
  if ( (unsigned int)v4 > 2 )
    goto LABEL_8;
  return sub_31B9240(a1, a2, a3, v4);
}
