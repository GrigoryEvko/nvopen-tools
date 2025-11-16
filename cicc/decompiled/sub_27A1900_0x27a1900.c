// Function: sub_27A1900
// Address: 0x27a1900
//
void __fastcall sub_27A1900(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // r14
  __int64 v4; // rbx

  if ( a2 - a1 <= 448 )
  {
    sub_27A1220(a1, a2);
  }
  else
  {
    v2 = (a2 - a1) >> 6;
    v3 = a1 + 32 * v2;
    v4 = (32 * v2) >> 5;
    sub_27A1900(a1, v3);
    sub_27A1900(v3, a2);
    sub_27A1770(a1, v3, a2, v4, (a2 - v3) >> 5);
  }
}
