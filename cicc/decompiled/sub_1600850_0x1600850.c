// Function: sub_1600850
// Address: 0x1600850
//
__int64 __fastcall sub_1600850(__int64 a1)
{
  __int16 v1; // cx
  _QWORD *v2; // r15
  unsigned int v3; // r14d
  unsigned int v4; // r13d
  unsigned int v5; // r14d
  __int64 v6; // rax
  __int64 v7; // r12
  __int16 v8; // dx
  __int16 v9; // ax
  __int16 v10; // dx
  __int64 v12; // [rsp+8h] [rbp-58h]
  char v13[16]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v14; // [rsp+20h] [rbp-40h]

  v1 = *(_WORD *)(a1 + 18);
  v2 = *(_QWORD **)(a1 + 56);
  v3 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
  v12 = *(_QWORD *)(a1 - 24);
  v14 = 257;
  v4 = (unsigned int)(1 << v1) >> 1;
  v5 = v3 >> 8;
  v6 = sub_1648A60(64, 1);
  v7 = v6;
  if ( v6 )
    sub_15F8A50(v6, v2, v5, v12, v4, (__int64)v13, 0);
  v8 = *(_WORD *)(v7 + 18) & 0x7FDF;
  if ( (*(_BYTE *)(a1 + 18) & 0x20) != 0 )
    v8 = *(_WORD *)(v7 + 18) & 0x7FDF | 0x20;
  v9 = v8 | *(_WORD *)(v7 + 18) & 0x8000;
  *(_WORD *)(v7 + 18) = v9;
  v10 = v9 & 0x7FBF;
  if ( (*(_BYTE *)(a1 + 18) & 0x40) != 0 )
    v10 |= 0x40u;
  *(_WORD *)(v7 + 18) = v10 | v9 & 0x8000;
  return v7;
}
