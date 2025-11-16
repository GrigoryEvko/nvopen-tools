// Function: sub_2147770
// Address: 0x2147770
//
unsigned __int64 __fastcall sub_2147770(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rsi
  __int64 v8; // rax
  unsigned __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // rax
  char v12; // dl
  __int64 v13; // rax
  bool v14; // al
  bool v15; // al
  __int64 v16; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // r8
  __int64 v19; // rax
  char v20; // dl
  __int64 v21; // rax
  bool v22; // al
  __int64 v23; // r9
  __int64 *v24; // rax
  __int64 v25; // rcx
  int v26; // edx
  const void ***v27; // rax
  __int64 v28; // r9
  __int64 *v29; // rax
  __int64 v30; // rsi
  unsigned int v31; // edx
  unsigned __int64 result; // rax
  bool v33; // al
  __int64 v34; // [rsp+8h] [rbp-C8h]
  __int64 v35; // [rsp+8h] [rbp-C8h]
  __int128 v36; // [rsp+40h] [rbp-90h] BYREF
  __int128 v37; // [rsp+50h] [rbp-80h] BYREF
  __int128 v38; // [rsp+60h] [rbp-70h] BYREF
  __int128 v39; // [rsp+70h] [rbp-60h] BYREF
  __int64 v40; // [rsp+80h] [rbp-50h] BYREF
  int v41; // [rsp+88h] [rbp-48h]
  _BYTE v42[8]; // [rsp+90h] [rbp-40h] BYREF
  __int64 v43; // [rsp+98h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 72);
  *(_QWORD *)&v36 = 0;
  DWORD2(v36) = 0;
  *(_QWORD *)&v37 = 0;
  DWORD2(v37) = 0;
  *(_QWORD *)&v38 = 0;
  DWORD2(v38) = 0;
  *(_QWORD *)&v39 = 0;
  DWORD2(v39) = 0;
  v40 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v40, v7, 2);
  v41 = *(_DWORD *)(a2 + 64);
  v8 = *(_QWORD *)(a2 + 32);
  v9 = *(_QWORD *)(v8 + 80);
  v10 = *(_QWORD *)(v8 + 88);
  v11 = *(_QWORD *)(v9 + 40) + 16LL * *(unsigned int *)(v8 + 88);
  v12 = *(_BYTE *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v42[0] = v12;
  v43 = v13;
  if ( v12 )
  {
    if ( (unsigned __int8)(v12 - 14) > 0x5Fu )
    {
      v14 = (unsigned __int8)(v12 - 2) <= 5u;
      goto LABEL_8;
    }
LABEL_18:
    sub_2017DE0(a1, v9, v10, &v36, &v37);
    goto LABEL_10;
  }
  v34 = v10;
  v15 = sub_1F58D20((__int64)v42);
  v10 = v34;
  if ( v15 )
    goto LABEL_18;
  v14 = sub_1F58CF0((__int64)v42);
  v10 = v34;
LABEL_8:
  if ( v14 )
    sub_20174B0(a1, v9, v10, &v36, &v37);
  else
    sub_2016B80(a1, v9, v10, &v36, &v37);
LABEL_10:
  v16 = *(_QWORD *)(a2 + 32);
  v17 = *(_QWORD *)(v16 + 120);
  v18 = *(_QWORD *)(v16 + 128);
  v19 = *(_QWORD *)(v17 + 40) + 16LL * *(unsigned int *)(v16 + 128);
  v20 = *(_BYTE *)v19;
  v21 = *(_QWORD *)(v19 + 8);
  v42[0] = v20;
  v43 = v21;
  if ( v20 )
  {
    if ( (unsigned __int8)(v20 - 14) > 0x5Fu )
    {
      v22 = (unsigned __int8)(v20 - 2) <= 5u;
      goto LABEL_13;
    }
  }
  else
  {
    v35 = v18;
    v33 = sub_1F58D20((__int64)v42);
    v18 = v35;
    if ( !v33 )
    {
      v22 = sub_1F58CF0((__int64)v42);
      v18 = v35;
LABEL_13:
      if ( v22 )
        sub_20174B0(a1, v17, v18, &v38, &v39);
      else
        sub_2016B80(a1, v17, v18, &v38, &v39);
      goto LABEL_15;
    }
  }
  sub_2017DE0(a1, v17, v18, &v38, &v39);
LABEL_15:
  v24 = sub_1D36A20(
          *(__int64 **)(a1 + 8),
          136,
          (__int64)&v40,
          *(unsigned __int8 *)(*(_QWORD *)(v36 + 40) + 16LL * DWORD2(v36)),
          *(const void ***)(*(_QWORD *)(v36 + 40) + 16LL * DWORD2(v36) + 8),
          v23,
          *(_OWORD *)*(_QWORD *)(a2 + 32),
          *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
          v36,
          v38,
          *(_OWORD *)(*(_QWORD *)(a2 + 32) + 160LL));
  v25 = v37;
  *(_QWORD *)a3 = v24;
  *(_DWORD *)(a3 + 8) = v26;
  v27 = (const void ***)(*(_QWORD *)(v25 + 40) + 16LL * DWORD2(v37));
  v29 = sub_1D36A20(
          *(__int64 **)(a1 + 8),
          136,
          (__int64)&v40,
          *(unsigned __int8 *)v27,
          v27[1],
          v28,
          *(_OWORD *)*(_QWORD *)(a2 + 32),
          *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
          v37,
          v39,
          *(_OWORD *)(*(_QWORD *)(a2 + 32) + 160LL));
  v30 = v40;
  *(_QWORD *)a4 = v29;
  result = v31;
  *(_DWORD *)(a4 + 8) = v31;
  if ( v30 )
    return sub_161E7C0((__int64)&v40, v30);
  return result;
}
