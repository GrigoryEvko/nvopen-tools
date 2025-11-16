// Function: sub_38D3050
// Address: 0x38d3050
//
__int64 __fastcall sub_38D3050(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 result; // rax
  _QWORD v21[2]; // [rsp+0h] [rbp-60h] BYREF
  char v22; // [rsp+10h] [rbp-50h]
  char v23; // [rsp+11h] [rbp-4Fh]
  _BYTE v24[16]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v25; // [rsp+30h] [rbp-30h]

  v2 = a1[86];
  v21[0] = ".text";
  v23 = 1;
  v22 = 3;
  v25 = 257;
  a1[3] = sub_38C2CC0(v2, (__int64)v21, 1u, (__int64)v24, -1);
  v3 = a1[86];
  v25 = 257;
  v23 = 1;
  v21[0] = ".data";
  v22 = 3;
  a1[4] = sub_38C2CC0(v3, (__int64)v21, 0x11u, (__int64)v24, -1);
  v4 = a1[86];
  v25 = 257;
  v23 = 1;
  v21[0] = ".debug_line";
  v22 = 3;
  a1[11] = sub_38C2CC0(v4, (__int64)v21, 0, (__int64)v24, -1);
  v5 = a1[86];
  v25 = 257;
  v23 = 1;
  v21[0] = ".debug_line_str";
  v22 = 3;
  a1[12] = sub_38C2CC0(v5, (__int64)v21, 0, (__int64)v24, -1);
  v6 = a1[86];
  v25 = 257;
  v23 = 1;
  v21[0] = ".debug_str";
  v22 = 3;
  a1[16] = sub_38C2CC0(v6, (__int64)v21, 0, (__int64)v24, -1);
  v7 = a1[86];
  v25 = 257;
  v23 = 1;
  v21[0] = ".debug_loc";
  v22 = 3;
  v8 = sub_38C2CC0(v7, (__int64)v21, 0, (__int64)v24, -1);
  v23 = 1;
  a1[17] = v8;
  v9 = a1[86];
  v25 = 257;
  v21[0] = ".debug_abbrev";
  v22 = 3;
  a1[9] = sub_38C2CC0(v9, (__int64)v21, 0, (__int64)v24, -1);
  v10 = a1[86];
  v25 = 257;
  v23 = 1;
  v21[0] = ".debug_aranges";
  v22 = 3;
  a1[18] = sub_38C2CC0(v10, (__int64)v21, 0, (__int64)v24, -1);
  v11 = a1[86];
  v21[0] = ".debug_ranges";
  v23 = 1;
  v22 = 3;
  v25 = 257;
  a1[19] = sub_38C2CC0(v11, (__int64)v21, 0, (__int64)v24, -1);
  v12 = a1[86];
  v21[0] = ".debug_macinfo";
  v23 = 1;
  v22 = 3;
  v25 = 257;
  a1[20] = sub_38C2CC0(v12, (__int64)v21, 0, (__int64)v24, -1);
  v13 = a1[86];
  v21[0] = ".debug_addr";
  v23 = 1;
  v22 = 3;
  v25 = 257;
  a1[35] = sub_38C2CC0(v13, (__int64)v21, 0, (__int64)v24, -1);
  v14 = a1[86];
  v21[0] = ".debug_cu_index";
  v23 = 1;
  v22 = 3;
  v25 = 257;
  a1[38] = sub_38C2CC0(v14, (__int64)v21, 0, (__int64)v24, -1);
  v15 = a1[86];
  v21[0] = ".debug_tu_index";
  v23 = 1;
  v22 = 3;
  v25 = 257;
  a1[39] = sub_38C2CC0(v15, (__int64)v21, 0, (__int64)v24, -1);
  v16 = a1[86];
  v25 = 257;
  v23 = 1;
  v21[0] = ".debug_info";
  v22 = 3;
  a1[10] = sub_38C2CC0(v16, (__int64)v21, 0, (__int64)v24, -1);
  v17 = a1[86];
  v25 = 257;
  v23 = 1;
  v21[0] = ".debug_frame";
  v22 = 3;
  a1[13] = sub_38C2CC0(v17, (__int64)v21, 0, (__int64)v24, -1);
  v18 = a1[86];
  v25 = 257;
  v23 = 1;
  v21[0] = ".debug_pubnames";
  v22 = 3;
  a1[21] = sub_38C2CC0(v18, (__int64)v21, 0, (__int64)v24, -1);
  v19 = a1[86];
  v25 = 257;
  v23 = 1;
  v21[0] = ".debug_pubtypes";
  v22 = 3;
  result = sub_38C2CC0(v19, (__int64)v21, 0, (__int64)v24, -1);
  a1[14] = result;
  return result;
}
