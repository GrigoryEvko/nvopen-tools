// Function: sub_B2FDE0
// Address: 0xb2fde0
//
__int64 __fastcall sub_B2FDE0(__int64 a1, _BYTE *a2)
{
  __int64 v2; // rsi
  int v3; // eax
  __int64 v5; // [rsp+0h] [rbp-30h] BYREF
  int v6; // [rsp+8h] [rbp-28h]
  __int64 v7; // [rsp+10h] [rbp-20h]
  int v8; // [rsp+18h] [rbp-18h]

  if ( ((unsigned __int8)(*a2 - 2) <= 1u || !*a2) && (a2[7] & 0x20) != 0 && (v2 = sub_B91C10(a2, 21)) != 0 )
  {
    sub_ABEA30((__int64)&v5, v2);
    v3 = v6;
    *(_BYTE *)(a1 + 32) = 1;
    *(_DWORD *)(a1 + 8) = v3;
    *(_QWORD *)a1 = v5;
    *(_DWORD *)(a1 + 24) = v8;
    *(_QWORD *)(a1 + 16) = v7;
    return a1;
  }
  else
  {
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
  }
}
