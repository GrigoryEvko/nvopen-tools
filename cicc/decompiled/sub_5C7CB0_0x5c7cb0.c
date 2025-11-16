// Function: sub_5C7CB0
// Address: 0x5c7cb0
//
__int64 __fastcall sub_5C7CB0(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rax
  __int64 v5; // r12
  char v6; // al
  _QWORD v7[3]; // [rsp+8h] [rbp-18h] BYREF

  v7[0] = a2;
  v3 = sub_5C7B50(a1, (__int64)v7, a3);
  if ( unk_4F06A80 | unk_4F06A7C | unk_4F06A78 )
  {
    sub_684B30(2665, a1 + 56);
    *(_BYTE *)(a1 + 8) = 0;
    return v7[0];
  }
  if ( !v3 )
    return v7[0];
  v5 = *(_QWORD *)(v3 + 168);
  v6 = *(_BYTE *)(v5 + 25);
  if ( v6 && v6 != 3 )
  {
    sub_684AA0(unk_4F077BC == 0 ? 8 : 5, 647, a1 + 56);
    *(_BYTE *)(v5 + 20) |= 0x20u;
    *(_BYTE *)(v5 + 25) = 3;
  }
  else
  {
    *(_BYTE *)(v5 + 20) |= 0x20u;
    *(_BYTE *)(v5 + 25) = 3;
  }
  return v7[0];
}
