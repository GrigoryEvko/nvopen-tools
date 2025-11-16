// Function: sub_685DD0
// Address: 0x685dd0
//
__int64 __fastcall sub_685DD0(__int64 a1, char a2, int *a3, __int64 a4)
{
  __int64 v5; // r12
  unsigned __int8 v7[33]; // [rsp+Fh] [rbp-21h] BYREF

  v5 = sub_7245B0(a1, a3, a4);
  if ( !v5 && sub_67C430(a2, *a3, v7) )
    sub_685AD0(v7[0], 1702, a1, a3);
  return v5;
}
