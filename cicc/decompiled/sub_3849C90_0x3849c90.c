// Function: sub_3849C90
// Address: 0x3849c90
//
void __fastcall sub_3849C90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rsi
  __int64 v8; // rax
  unsigned __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // rax
  __int16 v12; // dx
  __int64 v13; // rax
  bool v14; // al
  bool v15; // al
  __int64 v16; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // r8
  __int64 v19; // rax
  __int16 v20; // dx
  __int64 v21; // rax
  __int64 v22; // r9
  unsigned __int8 *v23; // rax
  __int64 v24; // rcx
  int v25; // edx
  unsigned __int16 *v26; // rax
  __int64 v27; // r9
  unsigned __int8 *v28; // rax
  __int64 v29; // rsi
  int v30; // edx
  bool v31; // al
  bool v32; // al
  __int64 v33; // [rsp+8h] [rbp-C8h]
  __int64 v34; // [rsp+8h] [rbp-C8h]
  __int128 v35; // [rsp+40h] [rbp-90h] BYREF
  __int128 v36; // [rsp+50h] [rbp-80h] BYREF
  __int128 v37; // [rsp+60h] [rbp-70h] BYREF
  __int128 v38; // [rsp+70h] [rbp-60h] BYREF
  __int64 v39; // [rsp+80h] [rbp-50h] BYREF
  int v40; // [rsp+88h] [rbp-48h]
  __int16 v41; // [rsp+90h] [rbp-40h] BYREF
  __int64 v42; // [rsp+98h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)&v35 = 0;
  DWORD2(v35) = 0;
  *(_QWORD *)&v36 = 0;
  DWORD2(v36) = 0;
  *(_QWORD *)&v37 = 0;
  DWORD2(v37) = 0;
  *(_QWORD *)&v38 = 0;
  DWORD2(v38) = 0;
  v39 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v39, v7, 1);
  v40 = *(_DWORD *)(a2 + 72);
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_QWORD *)(v8 + 80);
  v10 = *(_QWORD *)(v8 + 88);
  v11 = *(_QWORD *)(v9 + 48) + 16LL * *(unsigned int *)(v8 + 88);
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v41 = v12;
  v42 = v13;
  if ( v12 )
  {
    if ( (unsigned __int16)(v12 - 17) > 0xD3u )
    {
      if ( (unsigned __int16)(v12 - 2) <= 7u || (unsigned __int16)(v12 - 176) <= 0x1Fu )
        goto LABEL_6;
LABEL_9:
      sub_375E6F0(a1, v9, v10, (__int64)&v35, (__int64)&v36);
      goto LABEL_10;
    }
  }
  else
  {
    v33 = v10;
    v14 = sub_30070B0((__int64)&v41);
    v10 = v33;
    if ( !v14 )
    {
      v15 = sub_3007070((__int64)&v41);
      v10 = v33;
      if ( v15 )
      {
LABEL_6:
        sub_375E510(a1, v9, v10, (__int64)&v35, (__int64)&v36);
        goto LABEL_10;
      }
      goto LABEL_9;
    }
  }
  sub_375E8D0(a1, v9, v10, (__int64)&v35, (__int64)&v36);
LABEL_10:
  v16 = *(_QWORD *)(a2 + 40);
  v17 = *(_QWORD *)(v16 + 120);
  v18 = *(_QWORD *)(v16 + 128);
  v19 = *(_QWORD *)(v17 + 48) + 16LL * *(unsigned int *)(v16 + 128);
  v20 = *(_WORD *)v19;
  v21 = *(_QWORD *)(v19 + 8);
  v41 = v20;
  v42 = v21;
  if ( v20 )
  {
    if ( (unsigned __int16)(v20 - 17) > 0xD3u )
    {
      if ( (unsigned __int16)(v20 - 2) <= 7u || (unsigned __int16)(v20 - 176) <= 0x1Fu )
        goto LABEL_13;
LABEL_20:
      sub_375E6F0(a1, v17, v18, (__int64)&v37, (__int64)&v38);
      goto LABEL_14;
    }
  }
  else
  {
    v34 = v18;
    v31 = sub_30070B0((__int64)&v41);
    v18 = v34;
    if ( !v31 )
    {
      v32 = sub_3007070((__int64)&v41);
      v18 = v34;
      if ( v32 )
      {
LABEL_13:
        sub_375E510(a1, v17, v18, (__int64)&v37, (__int64)&v38);
        goto LABEL_14;
      }
      goto LABEL_20;
    }
  }
  sub_375E8D0(a1, v17, v18, (__int64)&v37, (__int64)&v38);
LABEL_14:
  v23 = sub_33FC1D0(
          *(_QWORD **)(a1 + 8),
          207,
          (__int64)&v39,
          *(unsigned __int16 *)(*(_QWORD *)(v35 + 48) + 16LL * DWORD2(v35)),
          *(_QWORD *)(*(_QWORD *)(v35 + 48) + 16LL * DWORD2(v35) + 8),
          v22,
          *(_OWORD *)*(_QWORD *)(a2 + 40),
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
          v35,
          v37,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 160LL));
  v24 = v36;
  *(_QWORD *)a3 = v23;
  *(_DWORD *)(a3 + 8) = v25;
  v26 = (unsigned __int16 *)(*(_QWORD *)(v24 + 48) + 16LL * DWORD2(v36));
  v28 = sub_33FC1D0(
          *(_QWORD **)(a1 + 8),
          207,
          (__int64)&v39,
          *v26,
          *((_QWORD *)v26 + 1),
          v27,
          *(_OWORD *)*(_QWORD *)(a2 + 40),
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
          v36,
          v38,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 160LL));
  v29 = v39;
  *(_QWORD *)a4 = v28;
  *(_DWORD *)(a4 + 8) = v30;
  if ( v29 )
    sub_B91220((__int64)&v39, v29);
}
