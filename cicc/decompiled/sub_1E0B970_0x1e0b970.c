// Function: sub_1E0B970
// Address: 0x1e0b970
//
__int64 __fastcall sub_1E0B970(__int64 a1, __int64 a2, int a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  unsigned __int8 v6; // bl
  int v7; // r13d
  __int64 v8; // r14
  int v9; // r15d
  unsigned int v10; // r12d
  __int128 v12; // [rsp-30h] [rbp-A0h]
  int v13; // [rsp+8h] [rbp-68h]
  __int64 v14; // [rsp+10h] [rbp-60h]
  __int64 v15; // [rsp+18h] [rbp-58h]
  char v16; // [rsp+30h] [rbp-40h]
  int v17; // [rsp+34h] [rbp-3Ch]

  v4 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*(_QWORD *)a2 & 4) != 0 )
  {
    v5 = v4 | 4;
    if ( v4 )
      LODWORD(v4) = *(_DWORD *)(v4 + 12);
  }
  else if ( v4 )
  {
    v5 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
    v4 = *(_QWORD *)v4;
    if ( *(_BYTE *)(v4 + 8) == 16 )
      v4 = **(_QWORD **)(v4 + 16);
    LODWORD(v4) = *(_DWORD *)(v4 + 8) >> 8;
  }
  else
  {
    LODWORD(v4) = 0;
    v5 = 4;
  }
  v6 = *(_BYTE *)(a2 + 37);
  v17 = v4;
  v7 = *(unsigned __int16 *)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 24);
  v9 = *(unsigned __int8 *)(a2 + 36);
  v13 = a3;
  v10 = (unsigned int)(1 << *(_WORD *)(a2 + 34)) >> 1;
  v14 = *(_QWORD *)(a2 + 64);
  v16 = 0;
  *((_QWORD *)&v12 + 1) = *(_QWORD *)(a2 + 8);
  *(_QWORD *)&v12 = v5;
  v15 = sub_145CBF0((__int64 *)(a1 + 120), 80, 16);
  sub_1E342C0(v15, v7, v8, v10, v13, v14, v12, v16, v9, v6 & 0xF, v6 >> 4);
  return v15;
}
