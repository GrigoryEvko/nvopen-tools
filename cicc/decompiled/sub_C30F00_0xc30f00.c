// Function: sub_C30F00
// Address: 0xc30f00
//
__int64 __fastcall sub_C30F00(__int64 a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 v4; // [rsp+8h] [rbp-18h] BYREF

  v2 = (__int64 *)(a1 + 168);
  v4 = a2;
  sub_CB2850(a1 + 168);
  if ( !(unsigned __int8)sub_CB2870(a1 + 168, 0) )
    return sub_CB1B70(v2);
  sub_CB05C0(v2);
  sub_C30A00(v2, &v4);
  sub_CB2220(v2);
  nullsub_173(v2);
  return sub_CB1B70(v2);
}
