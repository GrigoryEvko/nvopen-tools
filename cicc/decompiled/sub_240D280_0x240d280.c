// Function: sub_240D280
// Address: 0x240d280
//
__int64 __fastcall sub_240D280(__int64 *a1, __int64 **a2)
{
  __int64 *v4; // rsi
  unsigned __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rdx
  unsigned __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // rdx
  unsigned __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rcx
  __int64 result; // rax
  __int64 v43; // rdx
  __int64 v44[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *a2;
  v44[0] = 0;
  v5 = sub_A7A090(v44, v4, 1, 79);
  v6 = *a1;
  v7 = a1[27];
  v44[0] = v5;
  v8 = sub_BA8C10(v6, (__int64)"__dfsan_load_callback", 0x15u, v7, v5);
  v44[0] = 0;
  a1[50] = v9;
  a1[49] = v8;
  v10 = sub_A7A090(v44, *a2, 1, 79);
  v11 = *a1;
  v12 = a1[27];
  v44[0] = v10;
  v13 = sub_BA8C10(v11, (__int64)"__dfsan_store_callback", 0x16u, v12, v10);
  v14 = *a1;
  v15 = a1[28];
  a1[52] = v16;
  a1[51] = v13;
  v17 = sub_BA8CA0(v14, (__int64)"__dfsan_mem_transfer_callback", 0x1Du, v15);
  v44[0] = 0;
  a1[54] = v18;
  a1[53] = v17;
  v19 = sub_A7A090(v44, *a2, 1, 79);
  v20 = *a1;
  v21 = a1[26];
  v44[0] = v19;
  v22 = sub_BA8C10(v20, (__int64)"__dfsan_cmp_callback", 0x14u, v21, v19);
  v44[0] = 0;
  a1[64] = v23;
  a1[63] = v22;
  v24 = sub_A7A090(v44, *a2, 1, 79);
  v25 = *a1;
  v26 = a1[22];
  v44[0] = v24;
  v27 = sub_BA8C10(v25, (__int64)"__dfsan_conditional_callback", 0x1Cu, v26, v24);
  v44[0] = 0;
  a1[56] = v28;
  a1[55] = v27;
  v29 = sub_A7A090(v44, *a2, 1, 79);
  v30 = *a1;
  v31 = a1[23];
  v44[0] = v29;
  v32 = sub_BA8C10(v30, (__int64)"__dfsan_conditional_callback_origin", 0x23u, v31, v29);
  v44[0] = 0;
  a1[58] = v33;
  a1[57] = v32;
  v34 = sub_A7A090(v44, *a2, 1, 79);
  v35 = *a1;
  v36 = a1[24];
  v44[0] = v34;
  v37 = sub_BA8C10(v35, (__int64)"__dfsan_reaches_function_callback", 0x21u, v36, v34);
  v44[0] = 0;
  a1[60] = v38;
  a1[59] = v37;
  v39 = sub_A7A090(v44, *a2, 1, 79);
  v40 = *a1;
  v41 = a1[25];
  v44[0] = v39;
  result = sub_BA8C10(v40, (__int64)"__dfsan_reaches_function_callback_origin", 0x28u, v41, v39);
  a1[61] = result;
  a1[62] = v43;
  return result;
}
