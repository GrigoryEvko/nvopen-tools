// Function: sub_2900A00
// Address: 0x2900a00
//
__int64 __fastcall sub_2900A00(__int64 *a1, unsigned __int64 a2)
{
  __int64 v4; // r14
  __int64 v5; // rax
  _DWORD *v6; // rax
  __int64 *v7; // r13
  unsigned int v8; // r15d
  __int64 v9; // r10
  __int64 v10; // rdi
  __int64 (__fastcall *v11)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v12; // rax
  _BYTE *v13; // r13
  __int64 *v14; // rax
  _QWORD *v15; // rdi
  __int64 **v16; // r15
  __int64 v17; // rdi
  __int64 (__fastcall *v18)(__int64, unsigned int, _BYTE *, __int64); // rax
  _BYTE *v19; // r10
  unsigned int **v20; // rdi
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rsi
  unsigned int v25; // ecx
  __int64 *v26; // rdx
  __int64 v27; // r9
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rdx
  int v31; // r8d
  __int64 v32; // rax
  char v33; // al
  __int64 v34; // r10
  __int64 v35; // rdx
  int v36; // r12d
  __int64 *v37; // r15
  __int64 *v38; // rax
  __int64 v39; // r15
  __int64 i; // r12
  __int64 v41; // rdx
  unsigned int v42; // esi
  __int64 v43; // rax
  int v44; // edx
  int v45; // r10d
  int v46; // [rsp+8h] [rbp-A8h]
  __int64 v47; // [rsp+8h] [rbp-A8h]
  __int64 v48; // [rsp+8h] [rbp-A8h]
  __int64 v49; // [rsp+8h] [rbp-A8h]
  __int64 **v50; // [rsp+10h] [rbp-A0h]
  __int64 v51; // [rsp+10h] [rbp-A0h]
  __int64 *v52; // [rsp+18h] [rbp-98h]
  __int64 *v53; // [rsp+18h] [rbp-98h]
  __int64 v54; // [rsp+18h] [rbp-98h]
  _BYTE *v55; // [rsp+18h] [rbp-98h]
  _BYTE *v56; // [rsp+18h] [rbp-98h]
  _BYTE v57[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v58; // [rsp+40h] [rbp-70h]
  _BYTE v59[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v60; // [rsp+70h] [rbp-40h]

  if ( *(_BYTE *)a2 <= 0x15u )
  {
    v4 = sub_AC9EC0(*(__int64 ***)(a2 + 8));
    goto LABEL_3;
  }
  v22 = *a1;
  v23 = *(unsigned int *)(v22 + 24);
  v24 = *(_QWORD *)(v22 + 8);
  if ( !(_DWORD)v23 )
    goto LABEL_40;
  v25 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v26 = (__int64 *)(v24 + 16LL * v25);
  v27 = *v26;
  if ( *v26 != a2 )
  {
    v44 = 1;
    while ( v27 != -4096 )
    {
      v45 = v44 + 1;
      v25 = (v23 - 1) & (v44 + v25);
      v26 = (__int64 *)(v24 + 16LL * v25);
      v27 = *v26;
      if ( *v26 == a2 )
        goto LABEL_21;
      v44 = v45;
    }
LABEL_40:
    v28 = *(_QWORD *)(v22 + 32);
    goto LABEL_41;
  }
LABEL_21:
  v28 = *(_QWORD *)(v22 + 32);
  if ( v26 == (__int64 *)(v24 + 16 * v23) )
  {
LABEL_41:
    v29 = v28 + 16LL * *(unsigned int *)(v22 + 40);
    goto LABEL_23;
  }
  v29 = v28 + 16LL * *((unsigned int *)v26 + 2);
LABEL_23:
  v4 = *(_QWORD *)(v29 + 8);
LABEL_3:
  v5 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
    v5 = **(_QWORD **)(v5 + 16);
  v6 = sub_AE2980(a1[1], *(_DWORD *)(v5 + 8) >> 8);
  v7 = (__int64 *)a1[2];
  v8 = v6[1];
  v58 = 257;
  v52 = v7;
  v9 = sub_BCD140((_QWORD *)a1[3], v8);
  if ( v9 == *(_QWORD *)(v4 + 8) )
  {
    v13 = (_BYTE *)v4;
    goto LABEL_12;
  }
  v10 = v7[10];
  v11 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v10 + 120LL);
  if ( v11 != sub_920130 )
  {
    v51 = v9;
    v43 = v11(v10, 47u, (_BYTE *)v4, v9);
    v9 = v51;
    v13 = (_BYTE *)v43;
    goto LABEL_11;
  }
  if ( *(_BYTE *)v4 <= 0x15u )
  {
    v50 = (__int64 **)v9;
    if ( (unsigned __int8)sub_AC4810(0x2Fu) )
      v12 = sub_ADAB70(47, v4, v50, 0);
    else
      v12 = sub_AA93C0(0x2Fu, v4, (__int64)v50);
    v9 = (__int64)v50;
    v13 = (_BYTE *)v12;
LABEL_11:
    if ( v13 )
      goto LABEL_12;
  }
  v60 = 257;
  v13 = (_BYTE *)sub_B51D30(47, v4, v9, (__int64)v59, 0, 0);
  if ( (unsigned __int8)sub_920620((__int64)v13) )
  {
    v30 = v52[12];
    v31 = *((_DWORD *)v52 + 26);
    if ( v30 )
    {
      v46 = *((_DWORD *)v52 + 26);
      sub_B99FD0((__int64)v13, 3u, v30);
      v31 = v46;
    }
    sub_B45150((__int64)v13, v31);
  }
  (*(void (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v52[11] + 16LL))(
    v52[11],
    v13,
    v57,
    v52[7],
    v52[8]);
  v32 = *v52;
  v47 = *v52 + 16LL * *((unsigned int *)v52 + 2);
  if ( *v52 != v47 )
  {
    do
    {
      v54 = v32;
      sub_B99FD0((__int64)v13, *(_DWORD *)v32, *(_QWORD *)(v32 + 8));
      v32 = v54 + 16;
    }
    while ( v47 != v54 + 16 );
  }
LABEL_12:
  v14 = (__int64 *)a1[2];
  v15 = (_QWORD *)a1[3];
  v58 = 257;
  v53 = v14;
  v16 = (__int64 **)sub_BCD140(v15, v8);
  if ( v16 == *(__int64 ***)(a2 + 8) )
  {
    v19 = (_BYTE *)a2;
    goto LABEL_18;
  }
  v17 = v53[10];
  v18 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v17 + 120LL);
  if ( v18 != sub_920130 )
  {
    v19 = (_BYTE *)v18(v17, 47u, (_BYTE *)a2, (__int64)v16);
    goto LABEL_17;
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x2Fu) )
      v19 = (_BYTE *)sub_ADAB70(47, a2, v16, 0);
    else
      v19 = (_BYTE *)sub_AA93C0(0x2Fu, a2, (__int64)v16);
LABEL_17:
    if ( v19 )
      goto LABEL_18;
  }
  v60 = 257;
  v48 = sub_B51D30(47, a2, (__int64)v16, (__int64)v59, 0, 0);
  v33 = sub_920620(v48);
  v34 = v48;
  if ( v33 )
  {
    v35 = v53[12];
    v36 = *((_DWORD *)v53 + 26);
    if ( v35 )
    {
      sub_B99FD0(v48, 3u, v35);
      v34 = v48;
    }
    v49 = v34;
    sub_B45150(v34, v36);
    v34 = v49;
  }
  v37 = v53;
  v55 = (_BYTE *)v34;
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v37[11] + 16LL))(
    v37[11],
    v34,
    v57,
    v37[7],
    v37[8]);
  v38 = v37;
  v39 = *v37;
  v19 = v55;
  for ( i = v39 + 16LL * *((unsigned int *)v38 + 2); i != v39; v19 = v56 )
  {
    v41 = *(_QWORD *)(v39 + 8);
    v42 = *(_DWORD *)v39;
    v39 += 16;
    v56 = v19;
    sub_B99FD0((__int64)v19, v42, v41);
  }
LABEL_18:
  v20 = (unsigned int **)a1[2];
  v60 = 257;
  sub_929DE0(v20, v19, v13, (__int64)v59, 0, 0);
  return v4;
}
