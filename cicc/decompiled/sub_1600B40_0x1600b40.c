// Function: sub_1600B40
// Address: 0x1600b40
//
__int64 __fastcall sub_1600B40(__int64 a1)
{
  unsigned int v1; // r8d
  __int64 v2; // r15
  int v3; // r13d
  int v4; // r14d
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 *v8; // [rsp+0h] [rbp-40h]
  char v9; // [rsp+Ch] [rbp-34h]

  v1 = *(unsigned __int16 *)(a1 + 18);
  v2 = *(_QWORD *)(a1 - 48);
  v8 = *(__int64 **)(a1 - 24);
  v9 = *(_BYTE *)(a1 + 56);
  v3 = (v1 >> 2) & 7;
  v4 = (v1 >> 5) & 0x7FFFBFF;
  v5 = sub_1648A60(64, 2);
  v6 = v5;
  if ( v5 )
    sub_15F9C10(v5, v4, v2, v8, v3, v9, 0);
  *(_WORD *)(v6 + 18) = *(_WORD *)(a1 + 18) & 1 | *(_WORD *)(v6 + 18) & 0xFFFE;
  return v6;
}
