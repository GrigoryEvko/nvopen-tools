// Function: sub_22A83E0
// Address: 0x22a83e0
//
__int64 __fastcall sub_22A83E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = (0x6DB6DB6DB6DB6DB7LL * ((a2 - a1) >> 3) + 1) / 2;
  v10 = 56 * v6;
  v7 = a1 + 56 * v6;
  if ( v6 <= a4 )
  {
    sub_22A77A0(a1, a1 + 56 * v6, a3);
    sub_22A77A0(v7, a2, a3);
  }
  else
  {
    sub_22A83E0(a1, a1 + 56 * v6, a3);
    sub_22A83E0(v7, a2, a3);
  }
  sub_22A7AD0(a1, v7, a2, 0x6DB6DB6DB6DB6DB7LL * (v10 >> 3), 0x6DB6DB6DB6DB6DB7LL * ((a2 - v7) >> 3), a3, a4);
  return v9;
}
