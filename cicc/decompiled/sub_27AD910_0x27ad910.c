// Function: sub_27AD910
// Address: 0x27ad910
//
void __fastcall sub_27AD910(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx

  if ( a2 - a1 <= 1008 )
  {
    sub_27AC720(a1, a2, a3, a4, a5, a6);
  }
  else
  {
    v6 = 72 * ((__int64)(0x8E38E38E38E38E39LL * ((a2 - a1) >> 3)) >> 1);
    sub_27AD910(a1, a1 + v6);
    sub_27AD910(a1 + v6, a2);
    sub_27AD6A0(a1, a1 + v6, a2, 0x8E38E38E38E38E39LL * (v6 >> 3), 0x8E38E38E38E38E39LL * ((a2 - (a1 + v6)) >> 3));
  }
}
