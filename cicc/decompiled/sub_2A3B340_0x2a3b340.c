// Function: sub_2A3B340
// Address: 0x2a3b340
//
__int64 __fastcall sub_2A3B340(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned __int8 *v10; // r14
  __int64 (__fastcall *v11)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8); // rax
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdi
  unsigned __int8 *v15; // r14
  __int64 (__fastcall *v16)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char); // rax
  unsigned __int8 *v17; // r13
  __int64 v18; // rdi
  __int64 (__fastcall *v19)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  unsigned __int8 *v20; // r14
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rdi
  unsigned __int8 *v24; // r13
  __int64 (__fastcall *v25)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char); // rax
  unsigned __int8 *v26; // r15
  __int64 v27; // rdi
  __int64 (__fastcall *v28)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v29; // r12
  __int64 v31; // r13
  __int64 v32; // r14
  __int64 v33; // rdx
  unsigned int v34; // esi
  __int64 v35; // r13
  __int64 v36; // rbx
  __int64 v37; // r13
  __int64 v38; // rdx
  unsigned int v39; // esi
  __int64 v40; // r12
  __int64 v41; // r13
  __int64 v42; // rdx
  unsigned int v43; // esi
  __int64 v44; // r13
  __int64 v45; // r15
  __int64 v46; // rdx
  unsigned int v47; // esi
  __int64 v48; // r15
  __int64 i; // r14
  __int64 v50; // rdx
  unsigned int v51; // esi
  unsigned __int8 *v53; // [rsp+8h] [rbp-F8h]
  char v54[32]; // [rsp+10h] [rbp-F0h] BYREF
  __int16 v55; // [rsp+30h] [rbp-D0h]
  _BYTE v56[32]; // [rsp+40h] [rbp-C0h] BYREF
  __int16 v57; // [rsp+60h] [rbp-A0h]
  _BYTE v58[32]; // [rsp+70h] [rbp-90h] BYREF
  __int16 v59; // [rsp+90h] [rbp-70h]
  _BYTE v60[32]; // [rsp+A0h] [rbp-60h] BYREF
  __int16 v61; // [rsp+C0h] [rbp-40h]

  v5 = *(_QWORD *)(a2 + 8);
  v59 = 257;
  v6 = sub_AD64C0(v5, -1, 0);
  v7 = *(_QWORD *)(a2 + 8);
  v53 = (unsigned __int8 *)v6;
  v57 = 257;
  v55 = 257;
  v8 = sub_AD64C0(v7, 56, 0);
  v9 = a1[10];
  v10 = (unsigned __int8 *)v8;
  v11 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8))(*(_QWORD *)v9 + 24LL);
  if ( v11 != sub_920250 )
  {
    v12 = v11(v9, 27u, (_BYTE *)a2, v10, 0);
    goto LABEL_6;
  }
  if ( *(_BYTE *)a2 <= 0x15u && *v10 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(27) )
      v12 = sub_AD5570(27, a2, v10, 0, 0);
    else
      v12 = sub_AABE40(0x1Bu, (unsigned __int8 *)a2, v10);
LABEL_6:
    if ( v12 )
      goto LABEL_7;
  }
  v61 = 257;
  v12 = sub_B504D0(27, a2, (__int64)v10, (__int64)v60, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v12,
    v54,
    a1[7],
    a1[8]);
  v31 = *a1;
  v32 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v32 )
  {
    do
    {
      v33 = *(_QWORD *)(v31 + 8);
      v34 = *(_DWORD *)v31;
      v31 += 16;
      sub_B99FD0(v12, v34, v33);
    }
    while ( v32 != v31 );
  }
LABEL_7:
  v13 = sub_AD64C0(*(_QWORD *)(v12 + 8), 12, 0);
  v14 = a1[10];
  v15 = (unsigned __int8 *)v13;
  v16 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v14 + 32LL);
  if ( v16 != sub_9201A0 )
  {
    v17 = (unsigned __int8 *)v16(v14, 25u, (_BYTE *)v12, v15, 1u, 1);
    goto LABEL_12;
  }
  if ( *(_BYTE *)v12 <= 0x15u && *v15 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(25) )
      v17 = (unsigned __int8 *)sub_AD5570(25, v12, v15, 3, 0);
    else
      v17 = (unsigned __int8 *)sub_AABE40(0x19u, (unsigned __int8 *)v12, v15);
LABEL_12:
    if ( v17 )
      goto LABEL_13;
  }
  v61 = 257;
  v17 = (unsigned __int8 *)sub_B504D0(25, v12, (__int64)v15, (__int64)v60, 0, 0);
  (*(void (__fastcall **)(__int64, unsigned __int8 *, _BYTE *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v17,
    v56,
    a1[7],
    a1[8]);
  v48 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  for ( i = *a1; v48 != i; i += 16 )
  {
    v50 = *(_QWORD *)(i + 8);
    v51 = *(_DWORD *)i;
    sub_B99FD0((__int64)v17, v51, v50);
  }
  sub_B447F0(v17, 1);
  sub_B44850(v17, 1);
LABEL_13:
  v18 = a1[10];
  v19 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v18 + 16LL);
  if ( v19 != sub_9202E0 )
  {
    v20 = (unsigned __int8 *)v19(v18, 30u, v17, v53);
    goto LABEL_18;
  }
  if ( *v17 <= 0x15u && *v53 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(30) )
      v20 = (unsigned __int8 *)sub_AD5570(30, (__int64)v17, v53, 0, 0);
    else
      v20 = (unsigned __int8 *)sub_AABE40(0x1Eu, v17, v53);
LABEL_18:
    if ( v20 )
      goto LABEL_19;
  }
  v61 = 257;
  v20 = (unsigned __int8 *)sub_B504D0(30, (__int64)v17, (__int64)v53, (__int64)v60, 0, 0);
  (*(void (__fastcall **)(__int64, unsigned __int8 *, _BYTE *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v20,
    v58,
    a1[7],
    a1[8]);
  v44 = *a1;
  v45 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v45 )
  {
    do
    {
      v46 = *(_QWORD *)(v44 + 8);
      v47 = *(_DWORD *)v44;
      v44 += 16;
      sub_B99FD0((__int64)v20, v47, v46);
    }
    while ( v45 != v44 );
  }
LABEL_19:
  v21 = *(_QWORD *)(a2 + 8);
  v57 = 257;
  v59 = 257;
  v22 = sub_AD64C0(v21, a3, 0);
  v23 = a1[10];
  v24 = (unsigned __int8 *)v22;
  v25 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v23 + 32LL);
  if ( v25 != sub_9201A0 )
  {
    v26 = (unsigned __int8 *)v25(v23, 13u, (_BYTE *)a2, v24, 0, 0);
    goto LABEL_24;
  }
  if ( *(_BYTE *)a2 <= 0x15u && *v24 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(13) )
      v26 = (unsigned __int8 *)sub_AD5570(13, a2, v24, 0, 0);
    else
      v26 = (unsigned __int8 *)sub_AABE40(0xDu, (unsigned __int8 *)a2, v24);
LABEL_24:
    if ( v26 )
      goto LABEL_25;
  }
  v61 = 257;
  v26 = (unsigned __int8 *)sub_B504D0(13, a2, (__int64)v24, (__int64)v60, 0, 0);
  (*(void (__fastcall **)(__int64, unsigned __int8 *, _BYTE *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v26,
    v56,
    a1[7],
    a1[8]);
  v40 = *a1;
  v41 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v41 )
  {
    do
    {
      v42 = *(_QWORD *)(v40 + 8);
      v43 = *(_DWORD *)v40;
      v40 += 16;
      sub_B99FD0((__int64)v26, v43, v42);
    }
    while ( v41 != v40 );
  }
LABEL_25:
  v27 = a1[10];
  v28 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v27 + 16LL);
  if ( v28 != sub_9202E0 )
  {
    v29 = v28(v27, 28u, v26, v20);
    goto LABEL_30;
  }
  if ( *v26 <= 0x15u && *v20 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(28) )
      v29 = sub_AD5570(28, (__int64)v26, v20, 0, 0);
    else
      v29 = sub_AABE40(0x1Cu, v26, v20);
LABEL_30:
    if ( v29 )
      return v29;
  }
  v61 = 257;
  v29 = sub_B504D0(28, (__int64)v26, (__int64)v20, (__int64)v60, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v29,
    v58,
    a1[7],
    a1[8]);
  v35 = 16LL * *((unsigned int *)a1 + 2);
  v36 = *a1;
  v37 = v36 + v35;
  while ( v37 != v36 )
  {
    v38 = *(_QWORD *)(v36 + 8);
    v39 = *(_DWORD *)v36;
    v36 += 16;
    sub_B99FD0(v29, v39, v38);
  }
  return v29;
}
