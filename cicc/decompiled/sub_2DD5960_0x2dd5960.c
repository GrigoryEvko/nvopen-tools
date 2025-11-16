// Function: sub_2DD5960
// Address: 0x2dd5960
//
__int64 __fastcall sub_2DD5960(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // [rsp-10h] [rbp-50h]
  __int64 v15; // [rsp+8h] [rbp-38h]

  v8 = (__int64)(0xCCCCCCCCCCCCCCCDLL * ((a2 - a1) >> 4) + 1) / 2;
  v9 = 80 * v8;
  v15 = a1 + 80 * v8;
  if ( v8 <= a4 )
  {
    sub_2DD4610(a1, a1 + 80 * v8, a3, a4, a5, a6);
    sub_2DD4610(v15, a2, a3, v11, v12, v13);
  }
  else
  {
    sub_2DD5960(a1, a1 + 80 * v8, a3);
    sub_2DD5960(v15, a2, a3);
  }
  sub_2DD5010(a1, v15, a2, 0xCCCCCCCCCCCCCCCDLL * (v9 >> 4), 0xCCCCCCCCCCCCCCCDLL * ((a2 - v15) >> 4), a3, a4);
  return v14;
}
