// Function: sub_5EB950
// Address: 0x5eb950
//
__int64 __fastcall sub_5EB950(unsigned __int8 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int64 v5; // r13
  _DWORD v7[5]; // [rsp+Ch] [rbp-14h] BYREF

  v4 = a3;
  for ( v7[0] = 0; *(_BYTE *)(v4 + 140) == 12; v4 = *(_QWORD *)(v4 + 160) )
    ;
  v5 = sub_67D930(a1, a2, a4, v4);
  sub_5E6230(v4, 0, v5, 0x35Au, v7);
  if ( !v7[0] )
    sub_721090(v4);
  return sub_685910(v5);
}
