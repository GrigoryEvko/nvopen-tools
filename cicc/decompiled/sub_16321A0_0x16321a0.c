// Function: sub_16321A0
// Address: 0x16321a0
//
__int64 __fastcall sub_16321A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  result = sub_1632000(a1, a2, a3);
  if ( result )
  {
    if ( *(_BYTE *)(result + 16) )
      return 0;
  }
  return result;
}
