// Function: sub_3259570
// Address: 0x3259570
//
__int64 __fastcall sub_3259570(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // rbx
  _QWORD *v8; // r15
  __int64 (*v9)(); // rax
  __int64 v10; // r12
  __int64 v11; // rcx
  __int64 v12; // r8
  unsigned __int64 v13; // rax
  unsigned __int8 *v14; // r8
  void (*v15)(); // rax
  __int64 v16; // r15
  __int64 i; // r12
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // r15
  int v21; // r12d
  __int64 v22; // rax
  const char *v24; // rax
  __int64 v25; // rdx
  const char *v26; // r8
  __int64 v27; // r9
  __int64 v28; // r12
  unsigned __int64 v29; // rax
  __int64 v30; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v31; // [rsp+18h] [rbp-E8h]
  __int64 v32; // [rsp+18h] [rbp-E8h]
  char v33; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v34; // [rsp+20h] [rbp-E0h]
  unsigned __int8 *v35; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v36; // [rsp+28h] [rbp-D8h]
  __int64 v37; // [rsp+28h] [rbp-D8h]
  const char *v38; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v39; // [rsp+38h] [rbp-C8h]
  __int64 v40; // [rsp+40h] [rbp-C0h]
  __int64 v41; // [rsp+48h] [rbp-B8h]
  unsigned __int64 v42; // [rsp+50h] [rbp-B0h]
  __int64 v43; // [rsp+58h] [rbp-A8h]
  __int64 v44; // [rsp+60h] [rbp-A0h]
  int v45; // [rsp+68h] [rbp-98h]
  char v46; // [rsp+70h] [rbp-90h]
  int v47; // [rsp+74h] [rbp-8Ch]
  const char *v48; // [rsp+80h] [rbp-80h] BYREF
  __int64 v49; // [rsp+88h] [rbp-78h]
  __int64 v50; // [rsp+90h] [rbp-70h]
  __int64 v51; // [rsp+98h] [rbp-68h]
  unsigned __int64 v52; // [rsp+A0h] [rbp-60h]
  __int64 v53; // [rsp+A8h] [rbp-58h]
  __int64 v54; // [rsp+B0h] [rbp-50h]
  int v55; // [rsp+B8h] [rbp-48h]
  char v56; // [rsp+C0h] [rbp-40h]
  int v57; // [rsp+C4h] [rbp-3Ch]

  v5 = *(_QWORD *)(a1 + 8);
  v6 = a2[11];
  v33 = 0;
  v7 = *(_QWORD *)(v5 + 224);
  v8 = *(_QWORD **)(v5 + 216);
  v9 = *(__int64 (**)())(*(_QWORD *)v7 + 96LL);
  if ( v9 != sub_C13EE0 )
    v33 = ((__int64 (__fastcall *)(__int64))v9)(v7);
  if ( !*(_BYTE *)(a1 + 28) )
  {
    v24 = sub_BD5D20(*a2);
    if ( v25 && *v24 == 1 )
    {
      --v25;
      ++v24;
    }
    v49 = v25;
    LOWORD(v52) = 261;
    v48 = v24;
    v28 = sub_E6C770((__int64)v8, (__int64 *)&v48, v25, 261, v26, v27);
    v29 = sub_E81A90(*(int *)(v6 + 756), v8, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 272LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
      v28,
      v29);
  }
  LOWORD(v52) = 259;
  v48 = "lsda_begin";
  v10 = sub_E6C380((__int64)v8, (__int64 *)&v48, 1, a4, a5);
  v48 = "lsda_end";
  LOWORD(v52) = 259;
  v30 = sub_E6C380((__int64)v8, (__int64 *)&v48, 1, v11, v12);
  v31 = sub_3259000(a1, v30, v10);
  v13 = sub_E81A90(16, v8, 0, 0);
  v14 = (unsigned __int8 *)sub_E81A00(2, v31, v13, v8, 0);
  LOWORD(v52) = 259;
  v48 = "Number of call sites";
  if ( v33 )
  {
    v15 = *(void (**)())(*(_QWORD *)v7 + 120LL);
    if ( v15 != nullsub_98 )
    {
      v35 = v14;
      ((void (__fastcall *)(__int64, const char **, __int64))v15)(v7, &v48, 1);
      v14 = v35;
    }
  }
  sub_E9A5B0(v7, v14);
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v7 + 208LL))(v7, v10, 0);
  v16 = a2[41];
  for ( i = *(_QWORD *)(v16 + 8); a2 + 40 != (__int64 *)i; i = *(_QWORD *)(i + 8) )
  {
    if ( *(_BYTE *)(i + 235) )
      break;
  }
  v18 = *(_QWORD *)i;
  v19 = *(_QWORD *)(v16 + 56);
  v50 = i;
  v51 = i;
  v36 = v19;
  v48 = (const char *)v6;
  v52 = (v18 & 0xFFFFFFFFFFFFFFF8LL) + 48;
  v49 = 0;
  v56 = 0;
  v57 = -1;
  v53 = 0;
  v54 = 0;
  v55 = -1;
  sub_32588C0((__int64)&v48);
  v40 = v16;
  v41 = i;
  v20 = 0;
  v21 = -1;
  v42 = v36;
  v38 = (const char *)v6;
  v39 = 0;
  v46 = 0;
  v47 = -1;
  v43 = 0;
  v44 = 0;
  v45 = -1;
  sub_32588C0((__int64)&v38);
  v22 = v40;
  v48 = v38;
  v32 = v49;
  v49 = v39;
  v37 = v50;
  v51 = v41;
  v34 = v52;
  v52 = v42;
  v50 = v40;
  v53 = v43;
  v54 = v44;
  v55 = v45;
  v56 = v46;
  v57 = v47;
  while ( v22 != v37 || v52 != v34 || v49 != v32 )
  {
    if ( v21 != -1 )
      sub_32591C0(a1, v6, v20, v53, v21);
    v20 = v54;
    v21 = v55;
    sub_32588C0((__int64)&v48);
    v22 = v50;
  }
  return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v7 + 208LL))(v7, v30, 0);
}
