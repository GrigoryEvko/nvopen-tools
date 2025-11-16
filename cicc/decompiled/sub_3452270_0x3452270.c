// Function: sub_3452270
// Address: 0x3452270
//
unsigned __int8 *__fastcall sub_3452270(__int64 a1, __int64 a2, _QWORD *a3)
{
  unsigned __int16 *v5; // rcx
  unsigned __int16 v6; // dx
  signed int v7; // r15d
  __int64 *v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // r13
  __int64 v12; // r8
  __int64 v13; // r9
  __int128 v14; // rax
  __int64 v15; // r9
  unsigned __int8 *v16; // r12
  int v17; // eax
  __int64 v19; // rax
  __int128 v20; // [rsp-F8h] [rbp-F8h]
  __int128 v21; // [rsp-D8h] [rbp-D8h]
  __int64 v22; // [rsp-A0h] [rbp-A0h]
  __int128 v23; // [rsp-98h] [rbp-98h]
  __int64 v24; // [rsp-88h] [rbp-88h]
  __int64 v25; // [rsp-88h] [rbp-88h]
  __int64 v26; // [rsp-78h] [rbp-78h]
  __int128 v27; // [rsp-78h] [rbp-78h]
  unsigned int v28; // [rsp-68h] [rbp-68h]
  __int64 v29; // [rsp-68h] [rbp-68h]
  unsigned __int16 v30; // [rsp-58h] [rbp-58h] BYREF
  __int64 v31; // [rsp-50h] [rbp-50h]
  __int64 v32; // [rsp-48h] [rbp-48h] BYREF
  int v33; // [rsp-40h] [rbp-40h]

  if ( (*(_BYTE *)(a2 + 28) & 0x20) == 0 )
    return 0;
  v5 = *(unsigned __int16 **)(a2 + 48);
  v6 = *v5;
  v7 = 2 * (*(_DWORD *)(a2 + 24) == 279) + 18;
  v31 = *((_QWORD *)v5 + 1);
  v30 = v6;
  if ( ((*(_DWORD *)(a1 + 4 * ((v6 >> 3) + 36LL * v7 - v7) + 521536) >> (4 * (v6 & 7))) & 0xF) != 0 )
  {
    if ( !v6 )
      goto LABEL_4;
LABEL_13:
    if ( (unsigned __int16)(v6 - 17) > 0xD3u )
      goto LABEL_5;
    return 0;
  }
  v19 = 1;
  if ( v6 == 1 )
  {
LABEL_12:
    if ( (*(_BYTE *)(a1 + 500 * v19 + 6620) & 0xFB) == 0 )
      goto LABEL_5;
    goto LABEL_13;
  }
  if ( v6 )
  {
    v19 = v6;
    if ( !*(_QWORD *)(a1 + 8LL * v6 + 112) )
      goto LABEL_13;
    goto LABEL_12;
  }
LABEL_4:
  if ( sub_30070B0((__int64)&v30) )
    return 0;
LABEL_5:
  v8 = *(__int64 **)(a2 + 40);
  v9 = *(_QWORD *)(a2 + 80);
  v10 = *v8;
  v11 = *((unsigned int *)v8 + 2);
  v32 = v9;
  v12 = v8[5];
  v13 = *((unsigned int *)v8 + 12);
  if ( v9 )
  {
    v24 = v8[5];
    v26 = v10;
    v28 = *((_DWORD *)v8 + 12);
    sub_B96E90((__int64)&v32, v9, 1);
    v12 = v24;
    v10 = v26;
    v13 = v28;
  }
  v33 = *(_DWORD *)(a2 + 72);
  v25 = v10;
  *(_QWORD *)&v27 = v12;
  v29 = v10;
  *((_QWORD *)&v27 + 1) = v13;
  v22 = v10;
  *(_QWORD *)&v23 = v12;
  *((_QWORD *)&v23 + 1) = v13;
  *(_QWORD *)&v14 = sub_33ED040(a3, v7);
  *((_QWORD *)&v21 + 1) = v11;
  *(_QWORD *)&v21 = v29;
  *((_QWORD *)&v20 + 1) = v11;
  *(_QWORD *)&v20 = v25;
  v16 = sub_33FC1D0(
          a3,
          207,
          (__int64)&v32,
          *(unsigned __int16 *)(*(_QWORD *)(v22 + 48) + 16 * v11),
          *(_QWORD *)(*(_QWORD *)(v22 + 48) + 16 * v11 + 8),
          v15,
          v20,
          v27,
          v21,
          v23,
          v14);
  if ( v32 )
    sub_B91220((__int64)&v32, v32);
  v17 = *(_DWORD *)(a2 + 28);
  LOBYTE(v17) = v17 | 0x80;
  *((_DWORD *)v16 + 7) = v17;
  return v16;
}
