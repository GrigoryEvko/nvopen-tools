// Function: sub_377A5E0
// Address: 0x377a5e0
//
void __fastcall sub_377A5E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r11
  unsigned __int16 *v16; // rdx
  __int64 v17; // r9
  int v18; // edx
  _QWORD *v19; // r13
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdi
  unsigned __int16 *v23; // rdx
  unsigned __int8 *v24; // rax
  __int64 v25; // rsi
  int v26; // edx
  __int128 v27; // [rsp-10h] [rbp-E0h]
  __int128 v28; // [rsp-10h] [rbp-E0h]
  _QWORD *v29; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v30; // [rsp+10h] [rbp-C0h]
  __int64 v31; // [rsp+18h] [rbp-B8h]
  __int128 v32; // [rsp+40h] [rbp-90h] BYREF
  __int128 v33; // [rsp+50h] [rbp-80h] BYREF
  __int64 v34; // [rsp+60h] [rbp-70h] BYREF
  int v35; // [rsp+68h] [rbp-68h]
  __int64 v36[2]; // [rsp+70h] [rbp-60h] BYREF
  __int64 v37[10]; // [rsp+80h] [rbp-50h] BYREF

  v7 = *(unsigned __int64 **)(a2 + 40);
  DWORD2(v32) = 0;
  DWORD2(v33) = 0;
  *(_QWORD *)&v32 = 0;
  v8 = v7[1];
  *(_QWORD *)&v33 = 0;
  sub_375E8D0(a1, *v7, v8, (__int64)&v32, (__int64)&v33);
  v9 = *(_QWORD *)(a2 + 80);
  v34 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v34, v9, 1);
  v10 = *(_QWORD *)(a1 + 8);
  v35 = *(_DWORD *)(a2 + 72);
  v11 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL);
  v12 = *(_QWORD *)(v11 + 104);
  LOWORD(v11) = *(_WORD *)(v11 + 96);
  v36[1] = v12;
  LOWORD(v36[0]) = v11;
  sub_33D0340((__int64)v37, v10, v36);
  v30 = v37[3];
  v29 = *(_QWORD **)(a1 + 8);
  v31 = v37[2];
  v13 = sub_33F7D60(v29, v37[0], v37[1]);
  v15 = v14;
  v16 = (unsigned __int16 *)(*(_QWORD *)(v32 + 48) + 16LL * DWORD2(v32));
  *((_QWORD *)&v27 + 1) = v15;
  *(_QWORD *)&v27 = v13;
  *(_QWORD *)a3 = sub_3406EB0(v29, *(_DWORD *)(a2 + 24), (__int64)&v34, *v16, *((_QWORD *)v16 + 1), v17, v32, v27);
  *(_DWORD *)(a3 + 8) = v18;
  v19 = *(_QWORD **)(a1 + 8);
  v20 = sub_33F7D60(v19, v31, v30);
  v22 = v21;
  v23 = (unsigned __int16 *)(*(_QWORD *)(v33 + 48) + 16LL * DWORD2(v33));
  *((_QWORD *)&v28 + 1) = v22;
  *(_QWORD *)&v28 = v20;
  v24 = sub_3406EB0(
          v19,
          *(_DWORD *)(a2 + 24),
          (__int64)&v34,
          *v23,
          *((_QWORD *)v23 + 1),
          *(unsigned int *)(a2 + 24),
          v33,
          v28);
  v25 = v34;
  *(_QWORD *)a4 = v24;
  *(_DWORD *)(a4 + 8) = v26;
  if ( v25 )
    sub_B91220((__int64)&v34, v25);
}
