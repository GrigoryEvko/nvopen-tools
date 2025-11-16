// Function: sub_685EB0
// Address: 0x685eb0
//
__int64 __fastcall sub_685EB0(__int64 a1, __int64 a2, char a3, int a4)
{
  __int64 v6; // r12
  unsigned __int8 v8; // [rsp+7h] [rbp-39h] BYREF
  int v9[14]; // [rsp+8h] [rbp-38h] BYREF

  v6 = sub_7246B0(a1, a2, v9);
  if ( !v6 && sub_67C430(a3, v9[0], &v8) )
    sub_685AD0(v8, a4, a1, v9);
  return v6;
}
