// Function: sub_3136000
// Address: 0x3136000
//
__int64 __fastcall sub_3136000(_QWORD *a1, __int64 **a2)
{
  __int64 *v3; // r12
  __int64 v4; // rax
  __int64 *v5; // rdi
  __int64 v6; // rax
  __int64 *v7; // rdi
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 *v18; // rdi
  __int64 *v19; // rdi
  __int64 *v20; // rdi
  __int64 *v21; // rdi
  __int64 *v22; // rax
  __int64 *v23; // rdi
  __int64 *v24; // rdi
  __int64 v25; // rax
  __int64 *v26; // rdx
  __int64 *v27; // rdi
  __int64 *v28; // rax
  __int64 result; // rax
  __int64 *v30; // rdx
  __int64 *v31; // rcx
  __int64 *v32; // rax
  __int64 v33; // rax
  __int64 *v34; // rax
  __int64 *v35; // [rsp+0h] [rbp-90h] BYREF
  __int64 *v36; // [rsp+8h] [rbp-88h]
  __int64 *v37; // [rsp+10h] [rbp-80h]
  __int64 *v38; // [rsp+18h] [rbp-78h]
  __int64 *v39; // [rsp+20h] [rbp-70h]
  __int64 *v40; // [rsp+28h] [rbp-68h]
  __int64 *v41; // [rsp+30h] [rbp-60h]
  __int64 *v42; // [rsp+38h] [rbp-58h]
  __int64 *v43; // [rsp+40h] [rbp-50h]
  __int64 *v44; // [rsp+48h] [rbp-48h]
  __int64 v45; // [rsp+50h] [rbp-40h]
  __int64 v46; // [rsp+58h] [rbp-38h]
  __int64 *v47; // [rsp+60h] [rbp-30h]

  v3 = *a2;
  a1[325] = sub_BCB120(*a2);
  a1[326] = sub_BCB2A0(v3);
  a1[327] = sub_BCB2B0(v3);
  a1[328] = sub_BCB2C0(v3);
  a1[329] = sub_BCB2D0(v3);
  a1[330] = sub_BCB2E0(v3);
  a1[331] = sub_BCE3C0(v3, 0);
  a1[332] = sub_BCE3C0(v3, 0);
  a1[333] = sub_BCE3C0(v3, 0);
  a1[334] = sub_BCE3C0(v3, 0);
  a1[335] = sub_BCB170(v3);
  a1[336] = sub_AE4420((__int64)(a2 + 39), (__int64)v3, 0);
  a1[337] = sub_BCD140(v3, 0x3Fu);
  a1[338] = sub_BCE3C0(v3, 0);
  a1[339] = sub_BCE3C0(v3, 0);
  a1[340] = sub_BCE3C0(v3, 0);
  a1[341] = sub_BCE3C0(v3, 0);
  v4 = sub_BCE3C0(v3, 0);
  v5 = (__int64 *)a1[329];
  a1[342] = v4;
  a1[343] = sub_BCD420(v5, 8);
  v6 = sub_BCE3C0(v3, 0);
  v7 = (__int64 *)a1[329];
  a1[344] = v6;
  a1[345] = sub_BCD420(v7, 3);
  a1[346] = sub_BCE3C0(v3, 0);
  v8 = sub_BCBBB0(v3, (__int64)"struct.ident_t", 14);
  if ( !v8 )
  {
    v35 = (__int64 *)a1[329];
    v36 = v35;
    v37 = v35;
    v38 = v35;
    v39 = (__int64 *)a1[331];
    v8 = sub_BD0E80(v3, (__int64 *)&v35, 5, "struct.ident_t", 0xEu, 0);
  }
  a1[347] = v8;
  a1[348] = sub_BCE3C0(v3, 0);
  v9 = sub_BCBBB0(v3, (__int64)"struct.__tgt_kernel_arguments", 29);
  if ( !v9 )
  {
    v30 = (__int64 *)a1[329];
    v31 = (__int64 *)a1[334];
    v37 = (__int64 *)a1[339];
    v38 = v37;
    v41 = v37;
    v42 = v37;
    v32 = (__int64 *)a1[330];
    v35 = v30;
    v43 = v32;
    v44 = v32;
    v33 = a1[345];
    v36 = v30;
    v39 = v31;
    v40 = v31;
    v47 = v30;
    v45 = v33;
    v46 = v33;
    v9 = sub_BD0E80(v3, (__int64 *)&v35, 13, "struct.__tgt_kernel_arguments", 0x1Du, 0);
  }
  a1[349] = v9;
  a1[350] = sub_BCE3C0(v3, 0);
  v10 = sub_BCBBB0(v3, (__int64)"struct.__tgt_async_info", 23);
  if ( !v10 )
  {
    v35 = (__int64 *)a1[331];
    v10 = sub_BD0E80(v3, (__int64 *)&v35, 1, "struct.__tgt_async_info", 0x17u, 0);
  }
  a1[351] = v10;
  a1[352] = sub_BCE3C0(v3, 0);
  v11 = sub_BCBBB0(v3, (__int64)"struct.kmp_dep_info", 19);
  if ( !v11 )
  {
    v35 = (__int64 *)a1[336];
    v36 = v35;
    v37 = (__int64 *)a1[327];
    v11 = sub_BD0E80(v3, (__int64 *)&v35, 3, "struct.kmp_dep_info", 0x13u, 0);
  }
  a1[353] = v11;
  a1[354] = sub_BCE3C0(v3, 0);
  v12 = sub_BCBBB0(v3, (__int64)"struct.kmp_task_ompbuilder_t", 28);
  if ( !v12 )
  {
    v34 = (__int64 *)a1[338];
    v37 = (__int64 *)a1[329];
    v35 = v34;
    v36 = v34;
    v38 = v34;
    v39 = v34;
    v12 = sub_BD0E80(v3, (__int64 *)&v35, 5, "struct.kmp_task_ompbuilder_t", 0x1Cu, 0);
  }
  a1[355] = v12;
  a1[356] = sub_BCE3C0(v3, 0);
  v13 = sub_BCBBB0(v3, (__int64)"struct.ConfigurationEnvironmentTy", 33);
  if ( !v13 )
  {
    v35 = (__int64 *)a1[327];
    v36 = v35;
    v37 = v35;
    v38 = (__int64 *)a1[329];
    v39 = v38;
    v40 = v38;
    v41 = v38;
    v42 = v38;
    v43 = v38;
    v13 = sub_BD0E80(v3, (__int64 *)&v35, 9, "struct.ConfigurationEnvironmentTy", 0x21u, 0);
  }
  a1[357] = v13;
  a1[358] = sub_BCE3C0(v3, 0);
  v14 = sub_BCBBB0(v3, (__int64)"struct.DynamicEnvironmentTy", 27);
  if ( !v14 )
  {
    v35 = (__int64 *)a1[328];
    v14 = sub_BD0E80(v3, (__int64 *)&v35, 1, "struct.DynamicEnvironmentTy", 0x1Bu, 0);
  }
  a1[359] = v14;
  a1[360] = sub_BCE3C0(v3, 0);
  v15 = sub_BCBBB0(v3, (__int64)"struct.KernelEnvironmentTy", 26);
  if ( !v15 )
  {
    v35 = (__int64 *)a1[357];
    v36 = (__int64 *)a1[348];
    v37 = (__int64 *)a1[360];
    v15 = sub_BD0E80(v3, (__int64 *)&v35, 3, "struct.KernelEnvironmentTy", 0x1Au, 0);
  }
  a1[361] = v15;
  a1[362] = sub_BCE3C0(v3, 0);
  v16 = sub_BCBBB0(v3, (__int64)"struct.KernelLaunchEnvironmentTy", 32);
  if ( !v16 )
  {
    v35 = (__int64 *)a1[329];
    v36 = v35;
    v16 = sub_BD0E80(v3, (__int64 *)&v35, 2, "struct.KernelLaunchEnvironmentTy", 0x20u, 0);
  }
  a1[363] = v16;
  v17 = sub_BCE3C0(v3, 0);
  v18 = (__int64 *)a1[325];
  a1[364] = v17;
  v35 = (__int64 *)a1[333];
  v36 = v35;
  a1[365] = sub_BCF480(v18, &v35, 2, 1u);
  a1[366] = sub_BCE3C0(v3, 0);
  v19 = (__int64 *)a1[325];
  v35 = (__int64 *)a1[338];
  v36 = v35;
  a1[367] = sub_BCF480(v19, &v35, 2, 0);
  a1[368] = sub_BCE3C0(v3, 0);
  v20 = (__int64 *)a1[325];
  v35 = (__int64 *)a1[338];
  v36 = v35;
  a1[369] = sub_BCF480(v20, &v35, 2, 0);
  a1[370] = sub_BCE3C0(v3, 0);
  v35 = (__int64 *)a1[338];
  a1[371] = sub_BCF480(v35, &v35, 1, 0);
  a1[372] = sub_BCE3C0(v3, 0);
  v21 = (__int64 *)a1[325];
  v35 = (__int64 *)a1[338];
  a1[373] = sub_BCF480(v21, &v35, 1, 0);
  a1[374] = sub_BCE3C0(v3, 0);
  v35 = (__int64 *)a1[338];
  v36 = v35;
  a1[375] = sub_BCF480(v35, &v35, 2, 0);
  a1[376] = sub_BCE3C0(v3, 0);
  v22 = (__int64 *)a1[338];
  v35 = (__int64 *)a1[329];
  v36 = v22;
  a1[377] = sub_BCF480(v35, &v35, 2, 0);
  a1[378] = sub_BCE3C0(v3, 0);
  v23 = (__int64 *)a1[325];
  v35 = (__int64 *)a1[338];
  v36 = (__int64 *)a1[328];
  v37 = v36;
  v38 = v36;
  a1[379] = sub_BCF480(v23, &v35, 4, 0);
  a1[380] = sub_BCE3C0(v3, 0);
  v24 = (__int64 *)a1[325];
  v35 = (__int64 *)a1[338];
  v36 = (__int64 *)a1[329];
  a1[381] = sub_BCF480(v24, &v35, 2, 0);
  v25 = sub_BCE3C0(v3, 0);
  v26 = (__int64 *)a1[329];
  v27 = (__int64 *)a1[325];
  a1[382] = v25;
  v28 = (__int64 *)a1[338];
  v36 = v26;
  v35 = v28;
  v37 = v28;
  a1[383] = sub_BCF480(v27, &v35, 3, 0);
  result = sub_BCE3C0(v3, 0);
  a1[384] = result;
  return result;
}
