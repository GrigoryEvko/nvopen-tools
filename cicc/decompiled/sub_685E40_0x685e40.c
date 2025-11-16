// Function: sub_685E40
// Address: 0x685e40
//
__int64 __fastcall sub_685E40(char *a1, __int64 a2, __int64 a3, char a4, int a5)
{
  __int64 v7; // r12
  unsigned __int8 v9; // [rsp+7h] [rbp-39h] BYREF
  int v10[14]; // [rsp+8h] [rbp-38h] BYREF

  v7 = sub_724620(a1);
  if ( !v7 && sub_67C430(a4, v10[0], &v9) )
    sub_685AD0(v9, a5, (__int64)a1, v10);
  return v7;
}
