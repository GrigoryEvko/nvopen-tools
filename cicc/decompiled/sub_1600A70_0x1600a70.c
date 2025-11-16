// Function: sub_1600A70
// Address: 0x1600a70
//
__int64 __fastcall sub_1600A70(__int64 a1)
{
  __int64 v1; // r14
  unsigned int v2; // r8d
  char v3; // r15
  unsigned int v4; // r13d
  __int64 v5; // rax
  __int64 v6; // r12
  __int16 v7; // dx
  __int64 **v9; // [rsp+8h] [rbp-48h]
  __int64 v10; // [rsp+10h] [rbp-40h]
  int v11; // [rsp+1Ch] [rbp-34h]

  v1 = *(_QWORD *)(a1 - 72);
  v2 = *(unsigned __int16 *)(a1 + 18);
  v3 = *(_BYTE *)(a1 + 56);
  v9 = *(__int64 ***)(a1 - 48);
  v10 = *(_QWORD *)(a1 - 24);
  v4 = (unsigned __int8)v2 >> 5;
  v11 = (v2 >> 2) & 7;
  v5 = sub_1648A60(64, 3);
  v6 = v5;
  if ( v5 )
    sub_15F99E0(v5, v1, v9, v10, v11, v4, v3, 0);
  v7 = *(_WORD *)(a1 + 18) & 1 | *(_WORD *)(v6 + 18) & 0xFFFE;
  *(_WORD *)(v6 + 18) = v7;
  *(_WORD *)(v6 + 18) = v7 & 0x8000 | v7 & 0x7EFF | ((*(_BYTE *)(a1 + 19) & 1) << 8);
  return v6;
}
