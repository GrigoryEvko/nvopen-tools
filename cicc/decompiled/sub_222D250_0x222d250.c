// Function: sub_222D250
// Address: 0x222d250
//
__int64 __fastcall sub_222D250(__int64 a1, __int64 a2, int a3)
{
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  if ( a3 == 3 )
  {
    sub_2241130(a1, 0, 0, "No associated state", 19);
    return a1;
  }
  if ( a3 > 3 )
  {
    if ( a3 == 4 )
    {
      sub_2241130(a1, 0, 0, "Broken promise", 14);
      return a1;
    }
    goto LABEL_9;
  }
  if ( a3 != 1 )
  {
    if ( a3 == 2 )
    {
      sub_2241130(a1, 0, 0, "Promise already satisfied", 25);
      return a1;
    }
LABEL_9:
    sub_2241130(a1, 0, 0, "Unknown error", 13);
    return a1;
  }
  sub_2241130(a1, 0, 0, "Future already retrieved", 24);
  return a1;
}
