// Function: sub_39AD190
// Address: 0x39ad190
//
__int64 __fastcall sub_39AD190(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // rsi
  __int64 v6; // r12
  __int64 (*v7)(); // rax
  const char *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rbx
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned int *v15; // r12
  void (*v16)(); // rax
  __int64 *v17; // rax
  __int64 v18; // r15
  __int64 i; // r12
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // r15
  int v23; // r12d
  __int64 v24; // rax
  __int64 v26; // [rsp+8h] [rbp-F8h]
  __int64 v27; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v28; // [rsp+18h] [rbp-E8h]
  __int64 v29; // [rsp+18h] [rbp-E8h]
  char v30; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v31; // [rsp+20h] [rbp-E0h]
  __int64 v32; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v33; // [rsp+28h] [rbp-D8h]
  __int64 v34; // [rsp+28h] [rbp-D8h]
  const char *v35; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v36; // [rsp+38h] [rbp-C8h]
  __int64 v37; // [rsp+40h] [rbp-C0h]
  __int64 v38; // [rsp+48h] [rbp-B8h]
  unsigned __int64 v39; // [rsp+50h] [rbp-B0h]
  __int64 v40; // [rsp+58h] [rbp-A8h]
  __int64 v41; // [rsp+60h] [rbp-A0h]
  int v42; // [rsp+68h] [rbp-98h]
  char v43; // [rsp+70h] [rbp-90h]
  int v44; // [rsp+74h] [rbp-8Ch]
  const char *v45; // [rsp+80h] [rbp-80h] BYREF
  __int64 v46; // [rsp+88h] [rbp-78h]
  __int64 v47; // [rsp+90h] [rbp-70h]
  __int64 v48; // [rsp+98h] [rbp-68h]
  unsigned __int64 v49; // [rsp+A0h] [rbp-60h]
  __int64 v50; // [rsp+A8h] [rbp-58h]
  __int64 v51; // [rsp+B0h] [rbp-50h]
  int v52; // [rsp+B8h] [rbp-48h]
  char v53; // [rsp+C0h] [rbp-40h]
  int v54; // [rsp+C4h] [rbp-3Ch]

  v3 = *(_QWORD *)(a1 + 8);
  v4 = a2[11];
  v30 = 0;
  v5 = *(_QWORD *)(v3 + 256);
  v6 = *(_QWORD *)(v3 + 248);
  v27 = v5;
  v7 = *(__int64 (**)())(*(_QWORD *)v5 + 80LL);
  if ( v7 != sub_168DB50 )
    v30 = ((__int64 (__fastcall *)(__int64))v7)(v5);
  v8 = sub_1649960(*a2);
  v10 = (__int64)v8;
  if ( v9 && *v8 == 1 )
  {
    --v9;
    v10 = (__int64)(v8 + 1);
  }
  v11 = sub_38BF800(v6, v10, v9);
  v12 = sub_38CB470(*(int *)(v4 + 724), v6);
  (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 240LL))(
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
    v11,
    v12);
  LOWORD(v47) = 259;
  v45 = "lsda_begin";
  v13 = sub_38BF8E0(v6, (__int64)&v45, 1, 1);
  v45 = "lsda_end";
  v32 = v13;
  LOWORD(v47) = 259;
  v26 = sub_38BF8E0(v6, (__int64)&v45, 1, 1);
  v28 = sub_39ACC30(a1, v26, v32);
  v14 = sub_38CB470(16, v6);
  v15 = (unsigned int *)sub_38CB1F0(2, v28, v14, v6, 0);
  LOWORD(v47) = 259;
  v45 = "Number of call sites";
  if ( v30 )
  {
    v16 = *(void (**)())(*(_QWORD *)v27 + 104LL);
    if ( v16 != nullsub_580 )
      ((void (__fastcall *)(__int64, const char **, __int64))v16)(v27, &v45, 1);
  }
  sub_38DDD30(v27, v15);
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v27 + 176LL))(v27, v32, 0);
  v17 = a2 + 40;
  v18 = a2[41];
  for ( i = *(_QWORD *)(v18 + 8); v17 != (__int64 *)i; i = *(_QWORD *)(i + 8) )
  {
    if ( *(_BYTE *)(i + 183) )
      break;
  }
  v20 = *(_QWORD *)i;
  v21 = *(_QWORD *)(v18 + 32);
  v47 = i;
  v48 = i;
  v33 = v21;
  v45 = (const char *)v4;
  v49 = (v20 & 0xFFFFFFFFFFFFFFF8LL) + 24;
  v46 = 0;
  v53 = 0;
  v54 = -1;
  v50 = 0;
  v51 = 0;
  v52 = -1;
  sub_39AC5C0((__int64)&v45);
  v37 = v18;
  v38 = i;
  v22 = 0;
  v23 = -1;
  v39 = v33;
  v35 = (const char *)v4;
  v36 = 0;
  v43 = 0;
  v44 = -1;
  v40 = 0;
  v41 = 0;
  v42 = -1;
  sub_39AC5C0((__int64)&v35);
  v24 = v37;
  v45 = v35;
  v29 = v46;
  v46 = v36;
  v34 = v47;
  v48 = v38;
  v31 = v49;
  v49 = v39;
  v47 = v37;
  v50 = v40;
  v51 = v41;
  v52 = v42;
  v53 = v43;
  v54 = v44;
  while ( v34 != v24 || v49 != v31 || v46 != v29 )
  {
    if ( v23 != -1 )
      sub_39ACD80(a1, v4, v22, v50, v23);
    v22 = v51;
    v23 = v52;
    sub_39AC5C0((__int64)&v45);
    v24 = v47;
  }
  return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v27 + 176LL))(v27, v26, 0);
}
