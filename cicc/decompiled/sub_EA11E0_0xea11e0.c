// Function: sub_EA11E0
// Address: 0xea11e0
//
__int64 __fastcall sub_EA11E0(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rdx
  char v5; // bl
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 v8; // r13
  __int16 v9; // r12
  __int64 v10; // r9
  __int64 v11; // r8
  char v12; // di
  int v13; // r11d
  __int64 v14; // r10
  __int64 v15; // rax
  __int64 v17; // [rsp+0h] [rbp-38h]
  __int64 v18; // [rsp+8h] [rbp-30h]

  v18 = a1[28];
  v2 = (__int64 *)a1[25];
  v17 = a1[27];
  v3 = a1[26];
  v4 = v2[7];
  v5 = *((_BYTE *)v2 + 26);
  v6 = *v2;
  v7 = v2[1];
  v8 = v2[2];
  v9 = *((_WORD *)v2 + 12);
  v10 = v2[5];
  v11 = v2[8];
  v12 = *((_BYTE *)v2 + 72);
  v13 = *((_DWORD *)v2 + 7);
  v14 = v2[4];
  v15 = v2[6];
  *(_QWORD *)a2 = v6;
  *(_QWORD *)(a2 + 8) = v7;
  *(_QWORD *)(a2 + 16) = v8;
  *(_WORD *)(a2 + 24) = v9;
  *(_BYTE *)(a2 + 26) = v5;
  *(_DWORD *)(a2 + 28) = v13;
  *(_QWORD *)(a2 + 32) = v14;
  *(_QWORD *)(a2 + 48) = v15;
  *(_QWORD *)(a2 + 80) = v3;
  *(_QWORD *)(a2 + 40) = v10;
  *(_QWORD *)(a2 + 56) = v4;
  *(_QWORD *)(a2 + 64) = v11;
  *(_BYTE *)(a2 + 72) = v12;
  *(_QWORD *)(a2 + 88) = v17;
  *(_QWORD *)(a2 + 96) = v18;
  *(_QWORD *)(a2 + 104) = v4;
  return v18;
}
