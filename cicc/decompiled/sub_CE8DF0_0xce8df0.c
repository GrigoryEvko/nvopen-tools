// Function: sub_CE8DF0
// Address: 0xce8df0
//
__int64 __fastcall sub_CE8DF0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // [rsp+8h] [rbp-28h]
  __int64 v5; // [rsp+10h] [rbp-20h]

  if ( (unsigned __int8)sub_B2D620(a2, "nvvm.reqntid", 0xCu) )
  {
    sub_CE7350(a1, a2, "nvvm.reqntid", 0xCu);
  }
  else
  {
    v4 = sub_CE8CC0(a2);
    v5 = sub_CE8D00(a2);
    v3 = sub_CE8CC0(a2);
    sub_CE72B0(a1, v3, SBYTE4(v3), v5, SBYTE4(v5), v4, SBYTE4(v4));
  }
  return a1;
}
