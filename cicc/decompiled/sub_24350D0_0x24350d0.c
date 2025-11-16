// Function: sub_24350D0
// Address: 0x24350d0
//
__int64 __fastcall sub_24350D0(__int64 a1, __int64 *a2)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rdi
  unsigned __int8 *v7; // rbx
  __int64 (__fastcall *v8)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8); // rax
  __int64 v9; // r15
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned __int8 *v13; // r14
  __int64 (__fastcall *v14)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v15; // r13
  __int64 v17; // rbx
  __int64 v18; // r14
  __int64 v19; // rdx
  unsigned int v20; // esi
  __int64 v21; // rbx
  __int64 v22; // r12
  __int64 v23; // rdx
  unsigned int v24; // esi
  __int64 v25; // rax
  char v26[32]; // [rsp+10h] [rbp-C0h] BYREF
  __int16 v27; // [rsp+30h] [rbp-A0h]
  char v28[32]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v29; // [rsp+60h] [rbp-70h]
  const char *v30[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v31; // [rsp+90h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 528);
  if ( !v4 )
  {
    v25 = sub_2A3A780(a2);
    *(_QWORD *)(a1 + 528) = v25;
    v4 = v25;
  }
  v27 = 257;
  v5 = sub_AD64C0(*(_QWORD *)(v4 + 8), *(unsigned int *)(a1 + 176), 0);
  v6 = a2[10];
  v7 = (unsigned __int8 *)v5;
  v8 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8))(*(_QWORD *)v6 + 24LL);
  if ( v8 != sub_920250 )
  {
    v9 = v8(v6, 26u, (_BYTE *)v4, v7, 0);
    goto LABEL_8;
  }
  if ( *(_BYTE *)v4 <= 0x15u && *v7 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(26) )
      v9 = sub_AD5570(26, v4, v7, 0, 0);
    else
      v9 = sub_AABE40(0x1Au, (unsigned __int8 *)v4, v7);
LABEL_8:
    if ( v9 )
      goto LABEL_9;
  }
  v31 = 257;
  v9 = sub_B504D0(26, v4, (__int64)v7, (__int64)v30, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
    a2[11],
    v9,
    v26,
    a2[7],
    a2[8]);
  v17 = *a2;
  v18 = *a2 + 16LL * *((unsigned int *)a2 + 2);
  if ( *a2 != v18 )
  {
    do
    {
      v19 = *(_QWORD *)(v17 + 8);
      v20 = *(_DWORD *)v17;
      v17 += 16;
      sub_B99FD0(v9, v20, v19);
    }
    while ( v18 != v17 );
  }
LABEL_9:
  v10 = *(_QWORD *)(a1 + 184);
  if ( v10 == 255 )
    goto LABEL_17;
  v29 = 257;
  v11 = sub_AD64C0(*(_QWORD *)(v9 + 8), v10, 0);
  v12 = a2[10];
  v13 = (unsigned __int8 *)v11;
  v14 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v12 + 16LL);
  if ( v14 != sub_9202E0 )
  {
    v15 = v14(v12, 28u, (_BYTE *)v9, v13);
    goto LABEL_15;
  }
  if ( *(_BYTE *)v9 <= 0x15u && *v13 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(28) )
      v15 = sub_AD5570(28, v9, v13, 0, 0);
    else
      v15 = sub_AABE40(0x1Cu, (unsigned __int8 *)v9, v13);
LABEL_15:
    if ( v15 )
    {
LABEL_16:
      v9 = v15;
      goto LABEL_17;
    }
  }
  v31 = 257;
  v15 = sub_B504D0(28, v9, (__int64)v13, (__int64)v30, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
    a2[11],
    v15,
    v28,
    a2[7],
    a2[8]);
  v21 = *a2;
  v22 = *a2 + 16LL * *((unsigned int *)a2 + 2);
  if ( v21 == v22 )
    goto LABEL_16;
  do
  {
    v23 = *(_QWORD *)(v21 + 8);
    v24 = *(_DWORD *)v21;
    v21 += 16;
    sub_B99FD0(v15, v24, v23);
  }
  while ( v22 != v21 );
  v9 = v15;
LABEL_17:
  v30[0] = "hwasan.uar.tag";
  v31 = 259;
  sub_BD6B50((unsigned __int8 *)v9, v30);
  return v9;
}
