// Function: sub_5CA5F0
// Address: 0x5ca5f0
//
__int64 __fastcall sub_5CA5F0(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rax
  __int64 v5; // rax
  __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  v6 = a2;
  v3 = sub_5C7B50(a1, (__int64)&v6, a3);
  if ( unk_4F06A80 | unk_4F06A7C | unk_4F06A78 )
  {
    sub_684B30(2665, a1 + 56);
    *(_BYTE *)(a1 + 8) = 0;
    return v6;
  }
  if ( !v3 )
    return v6;
  v5 = *(_QWORD *)(v3 + 168);
  if ( *(_BYTE *)(v5 + 25) <= 1u )
  {
    *(_BYTE *)(v5 + 20) |= 0x20u;
    *(_BYTE *)(v5 + 25) = 1;
  }
  else
  {
    sub_684AA0(unk_4F077BC == 0 ? 8 : 5, 647, a1 + 56);
  }
  return v6;
}
