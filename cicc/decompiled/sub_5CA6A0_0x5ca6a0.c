// Function: sub_5CA6A0
// Address: 0x5ca6a0
//
__int64 __fastcall sub_5CA6A0(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // r12
  _QWORD v6[3]; // [rsp+8h] [rbp-18h] BYREF

  v6[0] = a2;
  v3 = sub_5C7B50(a1, (__int64)v6, a3);
  if ( unk_4F06A80 | unk_4F06A7C | unk_4F06A78 )
  {
    sub_684B30(2665, a1 + 56);
    *(_BYTE *)(a1 + 8) = 0;
    return v6[0];
  }
  if ( !v3 )
    return v6[0];
  v5 = *(_QWORD *)(v3 + 168);
  if ( (*(_BYTE *)(v5 + 25) & 0xFD) != 0 )
    sub_684AA0(unk_4F077BC == 0 ? 8 : 5, 647, a1 + 56);
  *(_BYTE *)(v5 + 20) |= 0x20u;
  result = v6[0];
  *(_BYTE *)(v5 + 25) = 2;
  return result;
}
