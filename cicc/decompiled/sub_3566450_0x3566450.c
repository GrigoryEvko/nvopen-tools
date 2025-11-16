// Function: sub_3566450
// Address: 0x3566450
//
__int64 __fastcall sub_3566450(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdi
  void (*v9)(); // rax
  __int64 v10; // r14
  __int64 v11; // r13
  __int64 v12; // rax
  int v13; // ebx
  unsigned __int64 i; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned int v20; // r13d
  __int64 v22; // rdi
  int v23; // [rsp+Ch] [rbp-1054h]
  __int64 v24; // [rsp+18h] [rbp-1048h] BYREF
  __int64 v25[6]; // [rsp+20h] [rbp-1040h] BYREF
  _BYTE v26[280]; // [rsp+50h] [rbp-1010h] BYREF
  _BYTE v27[3128]; // [rsp+168h] [rbp-EF8h] BYREF
  __int64 v28; // [rsp+DA0h] [rbp-2C0h]
  __int64 v29; // [rsp+DA8h] [rbp-2B8h]
  __int64 v30; // [rsp+DB0h] [rbp-2B0h]
  unsigned __int8 v31; // [rsp+DB8h] [rbp-2A8h]
  __int64 v32; // [rsp+DC0h] [rbp-2A0h]
  __int64 v33; // [rsp+DC8h] [rbp-298h]
  __int64 v34; // [rsp+DD0h] [rbp-290h]
  int v35; // [rsp+DD8h] [rbp-288h]
  __int64 v36; // [rsp+DE0h] [rbp-280h]
  _BYTE v37[416]; // [rsp+DE8h] [rbp-278h] BYREF
  __int64 v38; // [rsp+F88h] [rbp-D8h]
  __int64 v39; // [rsp+F90h] [rbp-D0h]
  __int64 v40; // [rsp+F98h] [rbp-C8h]
  __int64 v41; // [rsp+FA0h] [rbp-C0h]
  __int64 v42; // [rsp+FA8h] [rbp-B8h]
  __int64 v43; // [rsp+FB0h] [rbp-B0h]
  __int64 v44; // [rsp+FB8h] [rbp-A8h]
  _QWORD *v45; // [rsp+FC0h] [rbp-A0h]
  __int64 v46; // [rsp+FC8h] [rbp-98h]
  _QWORD v47[3]; // [rsp+FD0h] [rbp-90h] BYREF
  int v48; // [rsp+FE8h] [rbp-78h]
  __int64 v49; // [rsp+FF0h] [rbp-70h]
  __int64 v50; // [rsp+FF8h] [rbp-68h]
  __int64 v51; // [rsp+1000h] [rbp-60h]
  int v52; // [rsp+1008h] [rbp-58h]
  unsigned __int64 v53; // [rsp+1010h] [rbp-50h] BYREF
  char *v54; // [rsp+1018h] [rbp-48h]
  char *v55; // [rsp+1020h] [rbp-40h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_32:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_501EACC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_32;
  }
  v5 = *(_QWORD *)(a1 + 784);
  v23 = *(_DWORD *)(a1 + 572);
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_501EACC)
     + 200;
  sub_2F91670((__int64)v25, *(__int64 **)(a1 + 200), *(_QWORD *)(a1 + 216), 0);
  v28 = a1;
  v31 = 0;
  v25[0] = (__int64)&unk_4A39258;
  v34 = a1 + 248;
  v29 = 0;
  v30 = 0;
  v32 = a2;
  v33 = v6;
  v35 = v23;
  v36 = v5;
  sub_2F8FF00((__int64)v37, (__int64)v26, (__int64)v27);
  v38 = 0;
  v45 = v47;
  v7 = *(_QWORD *)(a1 + 200);
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v46 = 0;
  memset(v47, 0, sizeof(v47));
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v8 = *(_QWORD *)(v7 + 16);
  v9 = *(void (**)())(*(_QWORD *)v8 + 376LL);
  if ( v9 != nullsub_1719 )
    ((void (__fastcall *)(__int64, unsigned __int64 *))v9)(v8, &v53);
  if ( LOBYTE(qword_503E020[17]) )
  {
    v22 = sub_22077B0(8u);
    if ( v22 )
      *(_QWORD *)v22 = &unk_4A39300;
    v24 = v22;
    if ( v54 == v55 )
    {
      sub_2ECB480(&v53, v54, &v24);
      v22 = v24;
    }
    else
    {
      if ( v54 )
      {
        *(_QWORD *)v54 = v22;
        v54 += 8;
        goto LABEL_8;
      }
      v54 = (char *)8;
    }
    if ( v22 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v22 + 16LL))(v22);
  }
LABEL_8:
  v10 = **(_QWORD **)(a2 + 32);
  v11 = v10 + 48;
  sub_2F90C60((__int64)v25, v10);
  v12 = *(_QWORD *)(v10 + 56);
  if ( v12 == v10 + 48 )
  {
    v13 = 0;
  }
  else
  {
    v13 = 0;
    do
    {
      v12 = *(_QWORD *)(v12 + 8);
      ++v13;
    }
    while ( v12 != v11 );
  }
  for ( i = sub_2E313E0(v10); v11 != i; --v13 )
  {
    while ( 1 )
    {
      if ( !i )
        BUG();
      if ( (*(_BYTE *)i & 4) == 0 )
        break;
      i = *(_QWORD *)(i + 8);
      --v13;
      if ( v11 == i )
        goto LABEL_17;
    }
    while ( (*(_BYTE *)(i + 44) & 8) != 0 )
      i = *(_QWORD *)(i + 8);
    i = *(_QWORD *)(i + 8);
  }
LABEL_17:
  v15 = sub_2E313E0(v10);
  sub_2F90C80((__int64)v25, v10, *(_QWORD *)(v10 + 56), v15, v13);
  sub_3563190(v25);
  nullsub_1668();
  sub_35414A0((__int64)v25, v10, v16, v17, v18, v19);
  v20 = v31;
  sub_3542290((__int64)v25);
  return v20;
}
