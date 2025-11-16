// Function: sub_1F6D5D0
// Address: 0x1f6d5d0
//
__int64 __fastcall sub_1F6D5D0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rsi
  char v3; // di
  __int64 v4; // rax
  char v6[8]; // [rsp+0h] [rbp-10h] BYREF
  __int64 v7; // [rsp+8h] [rbp-8h]

  v2 = *(_QWORD *)(a1 + 40) + 16LL * a2;
  v3 = *(_BYTE *)v2;
  v4 = *(_QWORD *)(v2 + 8);
  v6[0] = v3;
  v7 = v4;
  if ( v3 )
    return sub_1F6C8D0(v3);
  else
    return sub_1F58D40((__int64)v6);
}
