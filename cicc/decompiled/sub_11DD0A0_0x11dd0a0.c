// Function: sub_11DD0A0
// Address: 0x11dd0a0
//
__int64 __fastcall sub_11DD0A0(__int64 a1, __int64 a2, unsigned int **a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // rax
  char v9; // bl
  _QWORD *v10; // rax
  __int64 v11; // r9
  __int64 v12; // r13
  __int64 v13; // rbx
  unsigned int *v14; // r14
  __int64 v15; // rdx
  unsigned int v16; // esi
  __int64 v17; // rdi
  __int64 (__fastcall *v18)(__int64, unsigned int, _BYTE *, __int64); // rax
  _BYTE *v19; // rbx
  __int64 v20; // rsi
  __int64 v21; // r13
  _BYTE *v22; // rax
  __int64 v23; // rax
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v28; // rax
  unsigned int *v29; // r12
  __int64 v30; // r14
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // [rsp-8h] [rbp-C8h]
  __int64 v36; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v37; // [rsp+20h] [rbp-A0h]
  __int64 **v38; // [rsp+28h] [rbp-98h]
  unsigned int **v39; // [rsp+28h] [rbp-98h]
  _BYTE v40[32]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v41; // [rsp+50h] [rbp-70h]
  _QWORD v42[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v43; // [rsp+80h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v36 = *(_QWORD *)(a1 - 32 * v4);
  v37 = *(_QWORD *)(a1 + 32 * (1 - v4));
  v5 = sub_BCB2B0(a3[9]);
  v6 = (__int64)a3[6];
  v41 = 257;
  v7 = v5;
  v38 = (__int64 **)v5;
  v8 = sub_AA4E30(v6);
  v9 = sub_AE5020(v8, v7);
  v43 = 257;
  v10 = sub_BD2C40(80, unk_3F10A14);
  v12 = (__int64)v10;
  if ( v10 )
  {
    sub_B4D190((__int64)v10, (__int64)v38, v36, (__int64)v42, 0, v9, 0, 0);
    v11 = v33;
  }
  (*(void (__fastcall **)(unsigned int *, __int64, _BYTE *, unsigned int *, unsigned int *, __int64))(*(_QWORD *)a3[11] + 16LL))(
    a3[11],
    v12,
    v40,
    a3[7],
    a3[8],
    v11);
  v13 = (__int64)&(*a3)[4 * *((unsigned int *)a3 + 2)];
  if ( *a3 != (unsigned int *)v13 )
  {
    v14 = *a3;
    do
    {
      v15 = *((_QWORD *)v14 + 1);
      v16 = *v14;
      v14 += 4;
      sub_B99FD0(v12, v16, v15);
    }
    while ( (unsigned int *)v13 != v14 );
  }
  v41 = 257;
  if ( v38 == *(__int64 ***)(v37 + 8) )
  {
    v19 = (_BYTE *)v37;
    goto LABEL_12;
  }
  v17 = (__int64)a3[10];
  v18 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v17 + 120LL);
  if ( v18 != sub_920130 )
  {
    v19 = (_BYTE *)v18(v17, 38u, (_BYTE *)v37, (__int64)v38);
    goto LABEL_11;
  }
  if ( *(_BYTE *)v37 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x26u) )
      v19 = (_BYTE *)sub_ADAB70(38, v37, v38, 0);
    else
      v19 = (_BYTE *)sub_AA93C0(0x26u, v37, (__int64)v38);
LABEL_11:
    if ( v19 )
      goto LABEL_12;
  }
  v43 = 257;
  v19 = (_BYTE *)sub_B51D30(38, v37, (__int64)v38, (__int64)v42, 0, 0);
  (*(void (__fastcall **)(unsigned int *, _BYTE *, _BYTE *, unsigned int *, unsigned int *))(*(_QWORD *)a3[11] + 16LL))(
    a3[11],
    v19,
    v40,
    a3[7],
    a3[8]);
  v28 = (__int64)&(*a3)[4 * *((unsigned int *)a3 + 2)];
  if ( *a3 != (unsigned int *)v28 )
  {
    v39 = a3;
    v29 = *a3;
    v30 = v28;
    do
    {
      v31 = *((_QWORD *)v29 + 1);
      v32 = *v29;
      v29 += 4;
      sub_B99FD0((__int64)v19, v32, v31);
    }
    while ( (unsigned int *)v30 != v29 );
    a3 = v39;
  }
LABEL_12:
  v20 = 32;
  v42[0] = "char0cmp";
  v43 = 259;
  v21 = sub_92B530(a3, 0x20u, v12, v19, (__int64)v42);
  if ( a2 )
  {
    v22 = (_BYTE *)sub_AD64C0(*(_QWORD *)(a2 + 8), 0, 0);
    v43 = 257;
    v23 = sub_92B530(a3, 0x21u, a2, v22, (__int64)v42);
    v43 = 257;
    v24 = v23;
    v25 = sub_AD6530(*(_QWORD *)(v21 + 8), 33);
    v20 = v24;
    v21 = sub_B36550(a3, v24, v21, v25, (__int64)v42, 0);
  }
  v26 = sub_AD6530(*(_QWORD *)(a1 + 8), v20);
  v43 = 257;
  return sub_B36550(a3, v21, v36, v26, (__int64)v42, 0);
}
