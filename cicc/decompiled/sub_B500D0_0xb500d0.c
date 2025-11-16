// Function: sub_B500D0
// Address: 0xb500d0
//
__int64 __fastcall sub_B500D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // [rsp+0h] [rbp-30h] BYREF
  __int64 v7; // [rsp+8h] [rbp-28h]

  sub_B43C20((__int64)&v6, 0);
  v2 = *(_QWORD *)(a2 - 32);
  sub_B44260(a1, *(_QWORD *)(a2 + 8), 64, 1u, v6, v7);
  if ( *(_QWORD *)(a1 - 32) )
  {
    v3 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v3;
    if ( v3 )
      *(_QWORD *)(v3 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = v2;
  if ( v2 )
  {
    v4 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(a1 - 24) = v4;
    if ( v4 )
      *(_QWORD *)(v4 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = v2 + 16;
    *(_QWORD *)(v2 + 16) = a1 - 32;
  }
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 80) = 0x400000000LL;
  if ( *(_DWORD *)(a2 + 80) )
    sub_B483A0(a1 + 72, a2 + 72);
  result = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1u;
  *(_BYTE *)(a1 + 1) = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1;
  return result;
}
