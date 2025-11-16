// Function: sub_CE8EA0
// Address: 0xce8ea0
//
__int64 __fastcall sub_CE8EA0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // [rsp+8h] [rbp-28h]
  __int64 v5; // [rsp+10h] [rbp-20h]

  if ( (unsigned __int8)sub_B2D620(a2, "nvvm.cluster_dim", 0x10u) )
  {
    sub_CE7350(a1, a2, "nvvm.cluster_dim", 0x10u);
  }
  else
  {
    v4 = sub_CE8C80(a2);
    v5 = sub_CE8C40(a2);
    v3 = sub_CE8C00(a2);
    sub_CE72B0(a1, v3, SBYTE4(v3), v5, SBYTE4(v5), v4, SBYTE4(v4));
  }
  return a1;
}
