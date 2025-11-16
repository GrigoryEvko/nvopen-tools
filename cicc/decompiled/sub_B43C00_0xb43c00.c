// Function: sub_B43C00
// Address: 0xb43c00
//
__int64 __fastcall sub_B43C00(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( a2 )
  {
    *(_QWORD *)a1 = a2 + 24;
    *(_WORD *)(a1 + 8) = 0;
    return 0;
  }
  else
  {
    *(_OWORD *)a1 = 0;
  }
  return result;
}
