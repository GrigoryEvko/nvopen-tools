// Function: sub_2B18180
// Address: 0x2b18180
//
void __fastcall sub_2B18180(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 v7; // rcx
  char **v8; // r14
  __int64 v9; // rbx

  v6 = a2 - a1;
  if ( a2 - a1 <= 896 )
  {
    sub_2B0F980(a1, a2, a3, v6, a5, a6);
  }
  else
  {
    v7 = v6 >> 7;
    v8 = (char **)(a1 + (v7 << 6));
    v9 = v7 << 6 >> 6;
    sub_2B18180(a1, v8);
    sub_2B18180(v8, a2);
    sub_2B17F60(a1, v8, a2, v9, (a2 - (__int64)v8) >> 6);
  }
}
