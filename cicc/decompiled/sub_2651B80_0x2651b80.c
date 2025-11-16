// Function: sub_2651B80
// Address: 0x2651b80
//
void __fastcall sub_2651B80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx

  if ( a2 - a1 <= 1008 )
  {
    sub_2651040(a1, a2, a3);
  }
  else
  {
    v4 = 72 * ((__int64)(0x8E38E38E38E38E39LL * ((a2 - a1) >> 3)) >> 1);
    sub_2651B80(a1, a1 + v4);
    sub_2651B80(a1 + v4, a2);
    sub_2651880(a1, a1 + v4, a2, 0x8E38E38E38E38E39LL * (v4 >> 3), 0x8E38E38E38E38E39LL * ((a2 - (a1 + v4)) >> 3), a3);
  }
}
