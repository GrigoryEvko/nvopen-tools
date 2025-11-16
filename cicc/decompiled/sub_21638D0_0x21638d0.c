// Function: sub_21638D0
// Address: 0x21638d0
//
__int64 __fastcall sub_21638D0(__int64 a1, __int16 ***a2)
{
  __int64 result; // rax

  result = a1;
  *(_QWORD *)a1 = a1 + 16;
  if ( a2 == &off_4A02620 )
  {
    *(_QWORD *)(a1 + 8) = 2;
    strcpy((char *)(a1 + 16), "%f");
  }
  else if ( a2 == &off_4A02760 )
  {
    *(_QWORD *)(a1 + 8) = 2;
    strcpy((char *)(a1 + 16), "%h");
  }
  else if ( a2 == &off_4A026A0 )
  {
    *(_WORD *)(a1 + 16) = 26661;
    *(_BYTE *)(a1 + 18) = 104;
    *(_QWORD *)(a1 + 8) = 3;
    *(_BYTE *)(a1 + 19) = 0;
  }
  else if ( a2 == &off_4A02520 )
  {
    *(_WORD *)(a1 + 16) = 26149;
    *(_BYTE *)(a1 + 18) = 100;
    *(_QWORD *)(a1 + 8) = 3;
    *(_BYTE *)(a1 + 19) = 0;
  }
  else if ( a2 == &off_4A02460 )
  {
    *(_WORD *)(a1 + 16) = 29221;
    *(_BYTE *)(a1 + 18) = 113;
    *(_QWORD *)(a1 + 8) = 3;
    *(_BYTE *)(a1 + 19) = 0;
  }
  else if ( a2 == &off_4A024A0 )
  {
    *(_WORD *)(a1 + 16) = 29221;
    *(_BYTE *)(a1 + 18) = 100;
    *(_QWORD *)(a1 + 8) = 3;
    *(_BYTE *)(a1 + 19) = 0;
  }
  else if ( a2 == &off_4A025A0 )
  {
    *(_QWORD *)(a1 + 8) = 2;
    strcpy((char *)(a1 + 16), "%r");
  }
  else if ( a2 == &off_4A02720 )
  {
    *(_WORD *)(a1 + 16) = 29221;
    *(_BYTE *)(a1 + 18) = 115;
    *(_QWORD *)(a1 + 8) = 3;
    *(_BYTE *)(a1 + 19) = 0;
  }
  else if ( a2 == &off_4A027A0 )
  {
    *(_QWORD *)(a1 + 8) = 2;
    strcpy((char *)(a1 + 16), "%p");
  }
  else if ( a2 == (__int16 ***)&off_4A026E0 )
  {
    strcpy((char *)(a1 + 16), "!Special!");
    *(_QWORD *)(a1 + 8) = 9;
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 8;
    strcpy((char *)(a1 + 16), "INTERNAL");
  }
  return result;
}
