// Function: sub_147A340
// Address: 0x147a340
//
__int64 __fastcall sub_147A340(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v5; // [rsp+8h] [rbp-28h] BYREF
  __int64 v6; // [rsp+10h] [rbp-20h] BYREF
  unsigned int v7[3]; // [rsp+1Ch] [rbp-14h] BYREF

  v7[0] = a2;
  v6 = a3;
  v5 = a4;
  sub_147DF40(a1, v7, &v6, &v5, 0);
  if ( (unsigned __int8)sub_147A100(a1, v7[0], v6, v5) || (unsigned __int8)sub_147AD20(a1, v7[0], v6, v5) )
    return 1;
  else
    return sub_1481140(a1, v7[0], v6, v5);
}
