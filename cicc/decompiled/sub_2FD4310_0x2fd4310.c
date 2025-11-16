// Function: sub_2FD4310
// Address: 0x2fd4310
//
__int64 __fastcall sub_2FD4310(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  void *v17; // rsi
  __int64 v18; // rax
  unsigned __int64 v19; // r8
  unsigned __int64 v20; // r9
  __int64 v21; // rdi
  __int64 v22; // r14
  __int64 (*v23)(); // rdx
  __int64 v24; // rax
  __int64 v25; // rcx
  _BYTE *v26; // rdx
  unsigned int v27; // r12d
  _QWORD v29[10]; // [rsp+0h] [rbp-1650h] BYREF
  char v30; // [rsp+50h] [rbp-1600h] BYREF
  char *v31; // [rsp+550h] [rbp-1100h]
  __int64 v32; // [rsp+558h] [rbp-10F8h]
  __int64 v33; // [rsp+560h] [rbp-10F0h]
  char v34; // [rsp+568h] [rbp-10E8h] BYREF
  char *v35; // [rsp+578h] [rbp-10D8h]
  __int64 v36; // [rsp+580h] [rbp-10D0h]
  char v37; // [rsp+588h] [rbp-10C8h] BYREF
  char *v38; // [rsp+5C8h] [rbp-1088h]
  __int64 v39; // [rsp+5D0h] [rbp-1080h]
  char v40; // [rsp+5D8h] [rbp-1078h] BYREF
  int *v41; // [rsp+668h] [rbp-FE8h]
  __int64 v42; // [rsp+670h] [rbp-FE0h]
  int v43; // [rsp+678h] [rbp-FD8h] BYREF
  char *v44; // [rsp+680h] [rbp-FD0h]
  __int64 v45; // [rsp+688h] [rbp-FC8h]
  char v46; // [rsp+690h] [rbp-FC0h] BYREF
  __int64 v47; // [rsp+720h] [rbp-F30h]
  __int64 v48; // [rsp+728h] [rbp-F28h]
  __int64 v49; // [rsp+730h] [rbp-F20h]
  char *v50; // [rsp+738h] [rbp-F18h]
  __int64 v51; // [rsp+740h] [rbp-F10h]
  char v52; // [rsp+748h] [rbp-F08h] BYREF
  _QWORD *v53; // [rsp+768h] [rbp-EE8h]
  __int64 v54; // [rsp+770h] [rbp-EE0h]
  _QWORD v55[4]; // [rsp+778h] [rbp-ED8h] BYREF
  _BYTE v56[3768]; // [rsp+798h] [rbp-EB8h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_26:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_501EB0C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_26;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_501EB0C);
  v7 = *(__int64 **)(a1 + 8);
  v8 = v6;
  v9 = v6 + 200;
  v10 = *v7;
  v11 = v7[1];
  if ( v10 == v11 )
LABEL_24:
    BUG();
  while ( *(_UNKNOWN **)v10 != &unk_501EC08 )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_24;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(*(_QWORD *)(v10 + 8), &unk_501EC08);
  v13 = *(__int64 **)(a1 + 8);
  v14 = v12 + 200;
  v15 = *v13;
  v16 = v13[1];
  if ( v15 == v16 )
LABEL_25:
    BUG();
  v17 = &unk_5025C1C;
  while ( *(_UNKNOWN **)v15 != &unk_5025C1C )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_25;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_5025C1C);
  v21 = *(_QWORD *)(a2 + 16);
  v22 = v18 + 200;
  v29[0] = *(_QWORD *)(a2 + 48);
  v23 = *(__int64 (**)())(*(_QWORD *)v21 + 128LL);
  v24 = 0;
  if ( v23 != sub_2DAC790 )
    v24 = ((__int64 (__fastcall *)(__int64))v23)(v21);
  v29[1] = v24;
  v31 = &v34;
  v35 = &v37;
  v38 = &v40;
  v39 = 0x200000000LL;
  v41 = &v43;
  v45 = 0x200000000LL;
  v42 = 0x200000001LL;
  v50 = &v52;
  v29[8] = &v30;
  v44 = &v46;
  v25 = 0x400000000LL;
  v53 = v55;
  v26 = v56;
  v29[2] = v9;
  v27 = 0;
  v29[4] = v22;
  v29[3] = v14;
  memset(&v29[5], 0, 24);
  v29[9] = 0x1000000000LL;
  v32 = 0;
  v33 = 16;
  v36 = 0x1000000000LL;
  v43 = -1;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v51 = 0x400000000LL;
  v54 = 0;
  v55[0] = 0;
  v55[1] = 1;
  v55[2] = v56;
  v55[3] = 0x1000000000LL;
  if ( *(_DWORD *)(v8 + 328) && !*(_BYTE *)(a2 + 341) )
  {
    v17 = (void *)a2;
    v27 = sub_2FD2820((__int64)v29, a2, (__int64)v56, 0x400000000LL, v19, v20);
  }
  sub_2FCFFF0((__int64)v29, (__int64)v17, (__int64)v26, v25, v19, v20);
  return v27;
}
