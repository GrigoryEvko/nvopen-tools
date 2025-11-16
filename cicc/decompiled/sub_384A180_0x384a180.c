// Function: sub_384A180
// Address: 0x384a180
//
void __fastcall sub_384A180(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rsi
  unsigned __int64 *v8; // rax
  unsigned __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // rax
  __int16 v12; // dx
  __int64 v13; // rax
  __int64 v14; // r9
  bool v15; // al
  bool v16; // al
  unsigned __int8 *v17; // rax
  __int64 v18; // rcx
  int v19; // edx
  unsigned __int16 *v20; // rax
  __int64 v21; // r9
  unsigned __int8 *v22; // rax
  __int64 v23; // rsi
  int v24; // edx
  __int64 v25; // [rsp+8h] [rbp-A8h]
  __int128 v26; // [rsp+40h] [rbp-70h] BYREF
  __int128 v27; // [rsp+50h] [rbp-60h] BYREF
  __int64 v28; // [rsp+60h] [rbp-50h] BYREF
  int v29; // [rsp+68h] [rbp-48h]
  __int16 v30; // [rsp+70h] [rbp-40h] BYREF
  __int64 v31; // [rsp+78h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)&v26 = 0;
  DWORD2(v26) = 0;
  *(_QWORD *)&v27 = 0;
  DWORD2(v27) = 0;
  v28 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v28, v7, 1);
  v29 = *(_DWORD *)(a2 + 72);
  v8 = *(unsigned __int64 **)(a2 + 40);
  v9 = *v8;
  v10 = v8[1];
  v11 = *(_QWORD *)(*v8 + 48) + 16LL * *((unsigned int *)v8 + 2);
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v30 = v12;
  v31 = v13;
  if ( v12 )
  {
    if ( (unsigned __int16)(v12 - 17) > 0xD3u )
    {
      if ( (unsigned __int16)(v12 - 2) <= 7u || (unsigned __int16)(v12 - 176) <= 0x1Fu )
        goto LABEL_6;
LABEL_9:
      sub_375E6F0(a1, v9, v10, (__int64)&v26, (__int64)&v27);
      goto LABEL_10;
    }
  }
  else
  {
    v25 = v10;
    v15 = sub_30070B0((__int64)&v30);
    v10 = v25;
    if ( !v15 )
    {
      v16 = sub_3007070((__int64)&v30);
      v10 = v25;
      if ( v16 )
      {
LABEL_6:
        sub_375E510(a1, v9, v10, (__int64)&v26, (__int64)&v27);
        goto LABEL_10;
      }
      goto LABEL_9;
    }
  }
  sub_375E8D0(a1, v9, v10, (__int64)&v26, (__int64)&v27);
LABEL_10:
  v17 = sub_3406EB0(
          *(_QWORD **)(a1 + 8),
          4u,
          (__int64)&v28,
          *(unsigned __int16 *)(*(_QWORD *)(v26 + 48) + 16LL * DWORD2(v26)),
          *(_QWORD *)(*(_QWORD *)(v26 + 48) + 16LL * DWORD2(v26) + 8),
          v14,
          v26,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  v18 = v27;
  *(_QWORD *)a3 = v17;
  *(_DWORD *)(a3 + 8) = v19;
  v20 = (unsigned __int16 *)(*(_QWORD *)(v18 + 48) + 16LL * DWORD2(v27));
  v22 = sub_3406EB0(
          *(_QWORD **)(a1 + 8),
          4u,
          (__int64)&v28,
          *v20,
          *((_QWORD *)v20 + 1),
          v21,
          v27,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  v23 = v28;
  *(_QWORD *)a4 = v22;
  *(_DWORD *)(a4 + 8) = v24;
  if ( v23 )
    sub_B91220((__int64)&v28, v23);
}
