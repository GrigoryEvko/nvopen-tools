// Function: sub_253C940
// Address: 0x253c940
//
__int64 *__fastcall sub_253C940(__int64 *a1, __int64 a2)
{
  __int16 v2; // ax
  __int16 v3; // dx

  v2 = *(_WORD *)(a2 + 96);
  if ( (v2 & 7) == 7 )
  {
    sub_253C590(a1, "known not-captured");
    return a1;
  }
  else
  {
    v3 = *(_WORD *)(a2 + 98);
    if ( (v3 & 7) == 7 )
    {
      sub_253C590(a1, "assumed not-captured");
      return a1;
    }
    else if ( (v2 & 3) == 3 )
    {
      sub_253C590(a1, "known not-captured-maybe-returned");
      return a1;
    }
    else
    {
      if ( (v3 & 3) == 3 )
        sub_253C590(a1, "assumed not-captured-maybe-returned");
      else
        sub_253C590(a1, "assumed-captured");
      return a1;
    }
  }
}
