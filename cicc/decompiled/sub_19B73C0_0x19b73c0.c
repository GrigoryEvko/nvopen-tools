// Function: sub_19B73C0
// Address: 0x19b73c0
//
__int64 __fastcall sub_19B73C0(int a1, int a2, int a3, int a4, int a5, int a6, int a7)
{
  int v7; // r15d
  bool v8; // r10
  int v9; // esi
  bool v10; // r11
  char v11; // dl
  char v12; // bl
  char v13; // r14
  char v14; // r13
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // rax
  bool v19; // [rsp+1h] [rbp-3Fh]
  bool v20; // [rsp+2h] [rbp-3Eh]
  bool v21; // [rsp+3h] [rbp-3Dh]
  bool v22; // [rsp+8h] [rbp-38h]
  bool v23; // [rsp+9h] [rbp-37h]
  bool v24; // [rsp+Ah] [rbp-36h]
  char v25; // [rsp+Bh] [rbp-35h]

  v7 = 0;
  if ( a2 != -1 )
    v7 = a2;
  v8 = a2 != -1;
  v9 = 0;
  if ( a3 != -1 )
    v9 = a3;
  v10 = a3 != -1;
  v11 = 0;
  if ( a4 != -1 )
  {
    v11 = 1;
    v20 = a4 != 0;
  }
  v12 = 0;
  if ( a5 != -1 )
  {
    v12 = 1;
    v21 = a5 != 0;
  }
  v13 = 0;
  if ( a6 != -1 )
  {
    v13 = 1;
    v22 = a6 != 0;
  }
  v14 = 0;
  if ( a7 != -1 )
  {
    v14 = 1;
    v19 = a7 != 0;
  }
  v23 = v10;
  v24 = v8;
  v25 = v11;
  v15 = sub_22077B0(184);
  v16 = v15;
  if ( v15 )
  {
    *(_QWORD *)(v15 + 8) = 0;
    *(_QWORD *)(v15 + 16) = &unk_4FB224C;
    *(_QWORD *)(v15 + 80) = v15 + 64;
    *(_QWORD *)(v15 + 88) = v15 + 64;
    *(_QWORD *)(v15 + 128) = v15 + 112;
    *(_QWORD *)(v15 + 136) = v15 + 112;
    *(_QWORD *)v15 = off_49F45F0;
    *(_DWORD *)(v15 + 24) = 2;
    *(_QWORD *)(v15 + 32) = 0;
    *(_QWORD *)(v15 + 40) = 0;
    *(_QWORD *)(v15 + 48) = 0;
    *(_DWORD *)(v15 + 64) = 0;
    *(_QWORD *)(v15 + 72) = 0;
    *(_QWORD *)(v15 + 96) = 0;
    *(_DWORD *)(v15 + 112) = 0;
    *(_QWORD *)(v15 + 120) = 0;
    *(_QWORD *)(v15 + 144) = 0;
    *(_BYTE *)(v15 + 152) = 0;
    *(_DWORD *)(v15 + 156) = a1;
    *(_BYTE *)(v15 + 164) = v23;
    if ( v23 )
      *(_DWORD *)(v15 + 160) = v9;
    *(_BYTE *)(v15 + 172) = v24;
    if ( v24 )
      *(_DWORD *)(v15 + 168) = v7;
    *(_BYTE *)(v15 + 177) = v25;
    if ( v25 )
      *(_BYTE *)(v15 + 176) = v20;
    *(_BYTE *)(v15 + 179) = v12;
    if ( v12 )
      *(_BYTE *)(v15 + 178) = v21;
    *(_BYTE *)(v15 + 181) = v13;
    if ( v13 )
      *(_BYTE *)(v15 + 180) = v22;
    *(_BYTE *)(v15 + 183) = v14;
    if ( v14 )
      *(_BYTE *)(v15 + 182) = v19;
    v17 = sub_163A1D0();
    sub_19B71A0(v17);
  }
  return v16;
}
