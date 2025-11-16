// Function: sub_28810B0
// Address: 0x28810b0
//
__int64 __fastcall sub_28810B0(int a1, char a2, char a3, int a4, int a5, int a6, int a7, int a8, int a9)
{
  int v9; // edx
  int v10; // ecx
  char v11; // r13
  char v12; // r12
  char v13; // bl
  char v14; // r15
  __int64 v15; // rax
  __int64 v16; // r14
  __int128 *v17; // rax
  bool v19; // [rsp+6h] [rbp-4Ah]
  bool v20; // [rsp+7h] [rbp-49h]
  int v21; // [rsp+8h] [rbp-48h]
  int v22; // [rsp+Ch] [rbp-44h]
  bool v23; // [rsp+10h] [rbp-40h]
  bool v24; // [rsp+11h] [rbp-3Fh]
  bool v25; // [rsp+12h] [rbp-3Eh]
  bool v26; // [rsp+13h] [rbp-3Dh]

  v9 = 0;
  if ( a4 != -1 )
    v9 = a4;
  v26 = a4 != -1;
  v10 = 0;
  if ( a5 != -1 )
    v10 = a5;
  v25 = a5 != -1;
  v11 = 0;
  if ( a6 != -1 )
  {
    v11 = 1;
    v23 = a6 != 0;
  }
  v12 = 0;
  if ( a7 != -1 )
  {
    v12 = 1;
    v24 = a7 != 0;
  }
  v13 = 0;
  if ( a8 != -1 )
  {
    v13 = 1;
    v20 = a8 != 0;
  }
  v14 = 0;
  if ( a9 != -1 )
  {
    v14 = 1;
    v19 = a9 != 0;
  }
  v21 = v10;
  v22 = v9;
  v15 = sub_22077B0(0xD8u);
  v16 = v15;
  if ( v15 )
  {
    *(_QWORD *)(v15 + 8) = 0;
    *(_QWORD *)(v15 + 16) = &unk_5001CAC;
    *(_QWORD *)(v15 + 56) = v15 + 104;
    *(_QWORD *)(v15 + 112) = v15 + 160;
    *(_QWORD *)v15 = off_4A21548;
    *(_DWORD *)(v15 + 180) = v21;
    *(_DWORD *)(v15 + 172) = a1;
    *(_DWORD *)(v15 + 88) = 1065353216;
    *(_BYTE *)(v15 + 176) = a2;
    *(_DWORD *)(v15 + 144) = 1065353216;
    *(_BYTE *)(v15 + 177) = a3;
    *(_DWORD *)(v15 + 24) = 1;
    *(_QWORD *)(v15 + 32) = 0;
    *(_QWORD *)(v15 + 40) = 0;
    *(_QWORD *)(v15 + 48) = 0;
    *(_QWORD *)(v15 + 64) = 1;
    *(_QWORD *)(v15 + 72) = 0;
    *(_QWORD *)(v15 + 80) = 0;
    *(_QWORD *)(v15 + 96) = 0;
    *(_QWORD *)(v15 + 104) = 0;
    *(_QWORD *)(v15 + 120) = 1;
    *(_QWORD *)(v15 + 128) = 0;
    *(_QWORD *)(v15 + 136) = 0;
    *(_QWORD *)(v15 + 152) = 0;
    *(_QWORD *)(v15 + 160) = 0;
    *(_BYTE *)(v15 + 168) = 0;
    *(_BYTE *)(v15 + 184) = v25;
    *(_BYTE *)(v15 + 197) = v11;
    *(_BYTE *)(v15 + 192) = v26;
    *(_DWORD *)(v15 + 188) = v22;
    *(_BYTE *)(v15 + 196) = v23;
    *(_BYTE *)(v15 + 199) = v12;
    *(_BYTE *)(v15 + 198) = v24;
    *(_BYTE *)(v15 + 201) = v13;
    *(_BYTE *)(v15 + 200) = v20;
    *(_BYTE *)(v15 + 203) = v14;
    *(_BYTE *)(v15 + 202) = v19;
    *(_BYTE *)(v15 + 205) = 0;
    *(_BYTE *)(v15 + 212) = 0;
    v17 = sub_BC2B00();
    sub_2880EB0((__int64)v17);
  }
  return v16;
}
