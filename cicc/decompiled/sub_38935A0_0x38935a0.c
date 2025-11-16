// Function: sub_38935A0
// Address: 0x38935a0
//
__int64 __fastcall sub_38935A0(__int64 a1, __int64 **a2)
{
  unsigned __int64 v3; // rsi
  const char *v4; // [rsp+0h] [rbp-30h] BYREF
  char v5; // [rsp+10h] [rbp-20h]
  char v6; // [rsp+11h] [rbp-1Fh]

  if ( (unsigned __int8)sub_1643460((__int64)*a2) )
    return sub_3893310(a1, a2);
  v3 = *(_QWORD *)(a1 + 56);
  v6 = 1;
  v4 = "invalid function return type";
  v5 = 3;
  return sub_38814C0(a1 + 8, v3, (__int64)&v4);
}
