// Function: sub_2477AC0
// Address: 0x2477ac0
//
void __fastcall sub_2477AC0(__int64 *a1, __int64 a2)
{
  __int64 *v3; // rdx
  _BYTE *v4; // r14
  __int64 v5; // rdx
  _BYTE *v6; // rax
  __int64 v7; // rax
  __int64 v8; // rsi
  _BYTE *v9; // r14
  _QWORD *v10; // rax
  _BYTE *v11; // rcx
  __int64 **v12; // rax
  _BYTE *v13; // r10
  __int64 (__fastcall *v14)(__int64, _BYTE *, _BYTE *, __int64, __int64); // rax
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 (__fastcall *v17)(__int64, _BYTE *, _BYTE *, __int64, __int64); // rax
  __int64 v18; // r14
  _QWORD *v19; // rax
  unsigned int *v20; // rbx
  unsigned int *v21; // r14
  __int64 v22; // rdx
  unsigned int v23; // esi
  _QWORD *v24; // rax
  unsigned int *v25; // rbx
  __int64 v26; // rax
  unsigned int *v27; // r15
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // rax
  _BYTE *v31; // [rsp+8h] [rbp-148h]
  __int64 v32; // [rsp+8h] [rbp-148h]
  _BYTE *v33; // [rsp+8h] [rbp-148h]
  _BYTE *v34; // [rsp+10h] [rbp-140h]
  int v35; // [rsp+2Ch] [rbp-124h] BYREF
  _BYTE v36[32]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v37; // [rsp+50h] [rbp-100h]
  _BYTE v38[32]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v39; // [rsp+80h] [rbp-D0h]
  unsigned int *v40; // [rsp+90h] [rbp-C0h] BYREF
  unsigned int v41; // [rsp+98h] [rbp-B8h]
  char v42; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v43; // [rsp+C8h] [rbp-88h]
  __int64 v44; // [rsp+D0h] [rbp-80h]
  __int64 v45; // [rsp+E0h] [rbp-70h]
  __int64 v46; // [rsp+E8h] [rbp-68h]
  void *v47; // [rsp+110h] [rbp-40h]

  sub_23D0AB0((__int64)&v40, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v4 = (_BYTE *)sub_246F3F0((__int64)a1, *v3);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v5 = *(_QWORD *)(a2 - 8);
  else
    v5 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v6 = (_BYTE *)sub_246F3F0((__int64)a1, *(_QWORD *)(v5 + 32));
  v39 = 257;
  v7 = sub_A82480(&v40, v4, v6, (__int64)v38);
  v39 = 257;
  v8 = *(_QWORD *)(v7 + 8);
  v9 = (_BYTE *)v7;
  v10 = sub_2463540(a1, v8);
  v11 = v10;
  if ( v10 )
    v11 = (_BYTE *)sub_AD6530((__int64)v10, v8);
  v34 = (_BYTE *)sub_92B530(&v40, 0x21u, (__int64)v9, v11, (__int64)v38);
  v12 = (__int64 **)sub_2463540(a1, *(_QWORD *)(a2 + 8));
  v35 = 0;
  v37 = 257;
  v13 = (_BYTE *)sub_ACADE0(v12);
  v14 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v45 + 88LL);
  if ( v14 != sub_9482E0 )
  {
    v33 = v13;
    v30 = v14(v45, v13, v9, (__int64)&v35, 1);
    v13 = v33;
    v16 = v30;
LABEL_11:
    if ( v16 )
      goto LABEL_12;
    goto LABEL_24;
  }
  if ( *v13 <= 0x15u && *v9 <= 0x15u )
  {
    v31 = v13;
    v15 = sub_AAAE30((__int64)v13, (__int64)v9, &v35, 1);
    v13 = v31;
    v16 = v15;
    goto LABEL_11;
  }
LABEL_24:
  v32 = (__int64)v13;
  v39 = 257;
  v19 = sub_BD2C40(104, unk_3F148BC);
  v16 = (__int64)v19;
  if ( v19 )
  {
    sub_B44260((__int64)v19, *(_QWORD *)(v32 + 8), 65, 2u, 0, 0);
    *(_QWORD *)(v16 + 72) = v16 + 88;
    *(_QWORD *)(v16 + 80) = 0x400000000LL;
    sub_B4FD20(v16, v32, (__int64)v9, &v35, 1, (__int64)v38);
  }
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v46 + 16LL))(v46, v16, v36, v43, v44);
  if ( v40 != &v40[4 * v41] )
  {
    v20 = v40;
    v21 = &v40[4 * v41];
    do
    {
      v22 = *((_QWORD *)v20 + 1);
      v23 = *v20;
      v20 += 4;
      sub_B99FD0(v16, v23, v22);
    }
    while ( v21 != v20 );
  }
LABEL_12:
  v35 = 1;
  v37 = 257;
  v17 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v45 + 88LL);
  if ( v17 != sub_9482E0 )
  {
    v18 = v17(v45, (_BYTE *)v16, v34, (__int64)&v35, 1);
LABEL_16:
    if ( v18 )
      goto LABEL_17;
    goto LABEL_30;
  }
  if ( *(_BYTE *)v16 <= 0x15u && *v34 <= 0x15u )
  {
    v18 = sub_AAAE30(v16, (__int64)v34, &v35, 1);
    goto LABEL_16;
  }
LABEL_30:
  v39 = 257;
  v24 = sub_BD2C40(104, unk_3F148BC);
  v18 = (__int64)v24;
  if ( v24 )
  {
    sub_B44260((__int64)v24, *(_QWORD *)(v16 + 8), 65, 2u, 0, 0);
    *(_QWORD *)(v18 + 72) = v18 + 88;
    *(_QWORD *)(v18 + 80) = 0x400000000LL;
    sub_B4FD20(v18, v16, (__int64)v34, &v35, 1, (__int64)v38);
  }
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v46 + 16LL))(v46, v18, v36, v43, v44);
  v25 = v40;
  v26 = 4LL * v41;
  v27 = &v40[v26];
  if ( v40 != &v40[v26] )
  {
    do
    {
      v28 = *((_QWORD *)v25 + 1);
      v29 = *v25;
      v25 += 4;
      sub_B99FD0(v18, v29, v28);
    }
    while ( v27 != v25 );
  }
LABEL_17:
  sub_246EF60((__int64)a1, a2, v18);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_2477350((__int64)a1, a2);
  nullsub_61();
  v47 = &unk_49DA100;
  nullsub_63();
  if ( v40 != (unsigned int *)&v42 )
    _libc_free((unsigned __int64)v40);
}
