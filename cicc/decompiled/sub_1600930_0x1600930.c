// Function: sub_1600930
// Address: 0x1600930
//
__int64 __fastcall sub_1600930(__int64 a1)
{
  unsigned int v1; // eax
  __int64 v2; // r13
  char v3; // r15
  unsigned __int8 v4; // r14
  int v5; // ebx
  __int64 v6; // r12
  unsigned int v8; // [rsp+Ch] [rbp-54h]
  _BYTE v9[16]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v10; // [rsp+20h] [rbp-40h]

  v10 = 257;
  v1 = *(unsigned __int16 *)(a1 + 18);
  v2 = *(_QWORD *)(a1 - 24);
  v3 = *(_BYTE *)(a1 + 56);
  v4 = *(_WORD *)(a1 + 18) & 1;
  v5 = (v1 >> 7) & 7;
  v8 = 1 << (v1 >> 1) >> 1;
  v6 = sub_1648A60(64, 1);
  if ( v6 )
    sub_15F8F80(v6, *(_QWORD *)(*(_QWORD *)v2 + 24LL), v2, (__int64)v9, v4, v8, v5, v3, 0);
  return v6;
}
