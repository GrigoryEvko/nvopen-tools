// Function: sub_20A1720
// Address: 0x20a1720
//
__int64 __fastcall sub_20A1720(__int64 a1, unsigned int a2)
{
  __int64 v2; // rsi
  __int64 v3; // rdx
  _BYTE v5[8]; // [rsp+0h] [rbp-10h] BYREF
  __int64 v6; // [rsp+8h] [rbp-8h]

  v2 = *(_QWORD *)(a1 + 40) + 16LL * a2;
  v3 = *(_QWORD *)(v2 + 8);
  v5[0] = *(_BYTE *)v2;
  v6 = v3;
  if ( v5[0] )
    return sub_1F3E310(v5);
  else
    return sub_1F58D40((__int64)v5);
}
