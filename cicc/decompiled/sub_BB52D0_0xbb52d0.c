// Function: sub_BB52D0
// Address: 0xbb52d0
//
__int64 __fastcall sub_BB52D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = a1;
  if ( *(_BYTE *)a2 == 5 && *(_WORD *)(a2 + 2) == 34 )
  {
    sub_AC51A0(a1, a2);
    return a1;
  }
  else
  {
    *(_BYTE *)(a1 + 32) = 0;
  }
  return result;
}
