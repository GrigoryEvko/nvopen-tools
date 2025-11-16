// Function: sub_1809430
// Address: 0x1809430
//
unsigned __int64 __fastcall sub_1809430(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // rdi
  __int64 v4; // r14
  __int64 *v5; // rax
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  bool v9; // zf
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 *v23; // rax
  __int64 v24; // rax
  __int64 v25; // r14
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 *v28; // rax
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 v31; // rax
  __int64 v32; // r14
  __int64 *v33; // rax
  __int64 v34; // rax
  __int64 v35; // r14
  __int64 v36; // rax
  __int64 v37; // r14
  __int64 *v38; // rax
  __int64 v39; // rax
  __int64 v40; // r13
  unsigned __int64 result; // rax
  __int64 v42; // [rsp+10h] [rbp-90h] BYREF
  __int64 v43; // [rsp+18h] [rbp-88h]
  __int64 v44; // [rsp+20h] [rbp-80h]
  __int64 v45[3]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v46; // [rsp+48h] [rbp-58h]
  __int64 v47; // [rsp+50h] [rbp-50h]
  int v48; // [rsp+58h] [rbp-48h]
  __int64 v49; // [rsp+60h] [rbp-40h]
  __int64 v50; // [rsp+68h] [rbp-38h]

  v3 = (_QWORD *)a1[27];
  v4 = a1[26];
  memset(v45, 0, sizeof(v45));
  v46 = v3;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v5 = (__int64 *)sub_1643270(v3);
  v42 = v4;
  v6 = sub_1644EA0(v5, &v42, 1, 0);
  v7 = sub_1632080(a2, (__int64)"__asan_before_dynamic_init", 26, v6, 0);
  v8 = sub_1B28080(v7);
  a1[38] = v8;
  v9 = (*(_BYTE *)(v8 + 32) & 0x30) == 0;
  *(_BYTE *)(v8 + 32) &= 0xF0u;
  if ( !v9 )
    *(_BYTE *)(v8 + 33) |= 0x40u;
  v10 = (__int64 *)sub_1643270(v46);
  v11 = sub_1644EA0(v10, &v42, 0, 0);
  v12 = sub_1632080(a2, (__int64)"__asan_after_dynamic_init", 25, v11, 0);
  v13 = sub_1B28080(v12);
  a1[39] = v13;
  v9 = (*(_BYTE *)(v13 + 32) & 0x30) == 0;
  *(_BYTE *)(v13 + 32) &= 0xF0u;
  if ( !v9 )
    *(_BYTE *)(v13 + 33) |= 0x40u;
  v14 = a1[26];
  v15 = (__int64 *)sub_1643270(v46);
  v16 = sub_18093A0(a2, (__int64)"__asan_register_globals", 23, 0, v15, v14, v14);
  v17 = sub_1B28080(v16);
  a1[40] = v17;
  v9 = (*(_BYTE *)(v17 + 32) & 0x30) == 0;
  *(_BYTE *)(v17 + 32) &= 0xF0u;
  if ( !v9 )
    *(_BYTE *)(v17 + 33) |= 0x40u;
  v18 = a1[26];
  v19 = (__int64 *)sub_1643270(v46);
  v20 = sub_18093A0(a2, (__int64)"__asan_unregister_globals", 25, 0, v19, v18, v18);
  v21 = sub_1B28080(v20);
  a1[41] = v21;
  v9 = (*(_BYTE *)(v21 + 32) & 0x30) == 0;
  *(_BYTE *)(v21 + 32) &= 0xF0u;
  if ( !v9 )
    *(_BYTE *)(v21 + 33) |= 0x40u;
  v22 = a1[26];
  v23 = (__int64 *)sub_1643270(v46);
  v42 = v22;
  v24 = sub_1644EA0(v23, &v42, 1, 0);
  v25 = sub_1632080(a2, (__int64)"__asan_register_image_globals", 29, v24, 0);
  v26 = sub_1B28080(v25);
  a1[42] = v26;
  v9 = (*(_BYTE *)(v26 + 32) & 0x30) == 0;
  *(_BYTE *)(v26 + 32) &= 0xF0u;
  if ( !v9 )
    *(_BYTE *)(v26 + 33) |= 0x40u;
  v27 = a1[26];
  v28 = (__int64 *)sub_1643270(v46);
  v42 = v27;
  v29 = sub_1644EA0(v28, &v42, 1, 0);
  v30 = sub_1632080(a2, (__int64)"__asan_unregister_image_globals", 31, v29, 0);
  v31 = sub_1B28080(v30);
  a1[43] = v31;
  v9 = (*(_BYTE *)(v31 + 32) & 0x30) == 0;
  *(_BYTE *)(v31 + 32) &= 0xF0u;
  if ( !v9 )
    *(_BYTE *)(v31 + 33) |= 0x40u;
  v32 = a1[26];
  v33 = (__int64 *)sub_1643270(v46);
  v42 = v32;
  v43 = v32;
  v44 = v32;
  v34 = sub_1644EA0(v33, &v42, 3, 0);
  v35 = sub_1632080(a2, (__int64)"__asan_register_elf_globals", 27, v34, 0);
  v36 = sub_1B28080(v35);
  a1[44] = v36;
  v9 = (*(_BYTE *)(v36 + 32) & 0x30) == 0;
  *(_BYTE *)(v36 + 32) &= 0xF0u;
  if ( !v9 )
    *(_BYTE *)(v36 + 33) |= 0x40u;
  v37 = a1[26];
  v38 = (__int64 *)sub_1643270(v46);
  v42 = v37;
  v43 = v37;
  v44 = v37;
  v39 = sub_1644EA0(v38, &v42, 3, 0);
  v40 = sub_1632080(a2, (__int64)"__asan_unregister_elf_globals", 29, v39, 0);
  result = sub_1B28080(v40);
  a1[45] = result;
  v9 = (*(_BYTE *)(result + 32) & 0x30) == 0;
  *(_BYTE *)(result + 32) &= 0xF0u;
  if ( !v9 )
    *(_BYTE *)(result + 33) |= 0x40u;
  if ( v45[0] )
    return sub_161E7C0((__int64)v45, v45[0]);
  return result;
}
