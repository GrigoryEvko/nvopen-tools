// Function: sub_E89240
// Address: 0xe89240
//
__int64 __fastcall sub_E89240(__int64 a1)
{
  _QWORD *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rdi
  __int64 result; // rax
  __int64 v14; // [rsp+18h] [rbp-18h]

  BYTE4(v14) = 0;
  v2 = (_QWORD *)sub_E6E320(*(_QWORD **)(a1 + 920), "..text..", 8, 2, 256, 1, v14);
  BYTE4(v14) = 0;
  *(_QWORD *)(a1 + 24) = v2;
  v3 = v2[19];
  *(_QWORD *)(v3 + 56) = byte_3F871B3;
  *(_QWORD *)(v3 + 64) = 0;
  *(_BYTE *)(v3 + 72) = 1;
  v2[20] = byte_3F871B3;
  v2[21] = 0;
  v4 = sub_E6E320(*(_QWORD **)(a1 + 920), ".data", 5, 19, 261, 1, v14);
  BYTE4(v14) = 0;
  *(_QWORD *)(a1 + 32) = v4;
  v5 = sub_E6E320(*(_QWORD **)(a1 + 920), ".rodata", 7, 4, 257, 1, v14);
  BYTE4(v14) = 0;
  *(_QWORD *)(a1 + 48) = v5;
  *(_BYTE *)(v5 + 32) = 2;
  v6 = sub_E6E320(*(_QWORD **)(a1 + 920), ".rodata.8", 9, 4, 257, 1, v14);
  BYTE4(v14) = 0;
  *(_QWORD *)(a1 + 800) = v6;
  *(_BYTE *)(v6 + 32) = 3;
  v7 = sub_E6E320(*(_QWORD **)(a1 + 920), ".rodata.16", 10, 4, 257, 1, v14);
  BYTE4(v14) = 0;
  *(_QWORD *)(a1 + 808) = v7;
  *(_BYTE *)(v7 + 32) = 4;
  v8 = sub_E6E320(*(_QWORD **)(a1 + 920), ".tdata", 6, 13, 276, 1, v14);
  BYTE4(v14) = 0;
  v9 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 424) = v8;
  v10 = sub_E6E320(v9, "TOC", 3, 19, 271, 0, v14);
  BYTE4(v14) = 0;
  *(_QWORD *)(a1 + 792) = v10;
  *(_BYTE *)(v10 + 32) = 2;
  v11 = sub_E6E320(*(_QWORD **)(a1 + 920), ".gcc_except_table", 17, 4, 257, 0, v14);
  BYTE4(v14) = 0;
  v12 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 56) = v11;
  *(_QWORD *)(a1 + 64) = sub_E6E320(v12, ".eh_info_table", 14, 19, 261, 0, v14);
  *(_QWORD *)(a1 + 80) = sub_E6E320(*(_QWORD **)(a1 + 920), ".dwabrev", 8, 0, 261, 1, 0x100060000LL);
  *(_QWORD *)(a1 + 88) = sub_E6E320(*(_QWORD **)(a1 + 920), ".dwinfo", 7, 0, 261, 1, 0x100010000LL);
  *(_QWORD *)(a1 + 96) = sub_E6E320(*(_QWORD **)(a1 + 920), ".dwline", 7, 0, 261, 1, 0x100020000LL);
  *(_QWORD *)(a1 + 112) = sub_E6E320(*(_QWORD **)(a1 + 920), ".dwframe", 8, 0, 261, 1, 0x1000A0000LL);
  *(_QWORD *)(a1 + 184) = sub_E6E320(*(_QWORD **)(a1 + 920), ".dwpbnms", 8, 0, 261, 1, 0x100030000LL);
  *(_QWORD *)(a1 + 120) = sub_E6E320(*(_QWORD **)(a1 + 920), ".dwpbtyp", 8, 0, 261, 1, 0x100040000LL);
  *(_QWORD *)(a1 + 136) = sub_E6E320(*(_QWORD **)(a1 + 920), ".dwstr", 6, 0, 261, 1, 0x100070000LL);
  *(_QWORD *)(a1 + 144) = sub_E6E320(*(_QWORD **)(a1 + 920), ".dwloc", 6, 0, 261, 1, 0x100090000LL);
  *(_QWORD *)(a1 + 152) = sub_E6E320(*(_QWORD **)(a1 + 920), ".dwarnge", 8, 0, 261, 1, 0x100050000LL);
  *(_QWORD *)(a1 + 160) = sub_E6E320(*(_QWORD **)(a1 + 920), ".dwrnges", 8, 0, 261, 1, 0x100080000LL);
  result = sub_E6E320(*(_QWORD **)(a1 + 920), ".dwmac", 6, 0, 261, 1, 0x1000B0000LL);
  *(_QWORD *)(a1 + 168) = result;
  return result;
}
