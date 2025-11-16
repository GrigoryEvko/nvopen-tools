// Function: sub_383C680
// Address: 0x383c680
//
unsigned __int8 *__fastcall sub_383C680(__int64 a1, __int64 a2, __m128i a3)
{
  __int16 *v5; // rax
  __int64 *v6; // r9
  __int16 v7; // r14
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r10
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v12; // r8
  unsigned __int16 v13; // r15
  int v14; // r9d
  __int64 *v15; // rdi
  __int64 v16; // rax
  unsigned __int8 v17; // dl
  unsigned __int8 *v18; // rax
  unsigned int v19; // r9d
  __int64 v20; // rsi
  __int64 v21; // rbx
  unsigned int v22; // edx
  unsigned __int16 *v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // rdi
  unsigned __int8 *v28; // r14
  bool v30; // al
  __int64 v31; // r14
  int v32; // eax
  __int64 v33; // rdx
  __int64 v34; // [rsp+0h] [rbp-80h]
  int v35; // [rsp+0h] [rbp-80h]
  __int16 v36; // [rsp+2h] [rbp-7Eh]
  __int64 v37; // [rsp+18h] [rbp-68h]
  __int64 v38; // [rsp+18h] [rbp-68h]
  __int64 v39; // [rsp+20h] [rbp-60h] BYREF
  __int64 v40; // [rsp+28h] [rbp-58h]
  __int64 v41; // [rsp+30h] [rbp-50h] BYREF
  int v42; // [rsp+38h] [rbp-48h]
  __int64 v43; // [rsp+40h] [rbp-40h]

  v5 = *(__int16 **)(a2 + 48);
  v6 = *(__int64 **)a1;
  v7 = *v5;
  v8 = *((_QWORD *)v5 + 1);
  v9 = **(_QWORD **)a1;
  v10 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL);
  LOWORD(v39) = v7;
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(v9 + 592);
  v40 = v8;
  if ( v11 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v41, (__int64)v6, v10, v39, v40);
    v13 = v42;
    v37 = v43;
    v14 = (unsigned __int16)v42;
  }
  else
  {
    v32 = ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD))v11)(v6, v10, (unsigned int)v39);
    v37 = v33;
    v14 = v32;
    v13 = v32;
  }
  if ( v7 )
  {
    if ( (unsigned __int16)(v7 - 17) <= 0xD3u )
      goto LABEL_7;
    v15 = *(__int64 **)a1;
    v16 = 1;
    if ( v13 == 1 )
      goto LABEL_6;
LABEL_15:
    if ( !v13 )
      goto LABEL_17;
    v16 = v13;
    if ( !v15[v13 + 14] )
      goto LABEL_17;
    goto LABEL_6;
  }
  v35 = v14;
  v30 = sub_30070B0((__int64)&v39);
  v14 = v35;
  if ( v30 )
    goto LABEL_7;
  v15 = *(__int64 **)a1;
  v16 = 1;
  if ( v13 != 1 )
    goto LABEL_15;
LABEL_6:
  v17 = *((_BYTE *)v15 + 500 * (unsigned int)v16 + 6603);
  if ( v17 <= 1u || v17 == 4 )
    goto LABEL_7;
  if ( v13 == 1 )
  {
    v16 = 1;
    goto LABEL_24;
  }
  if ( v15[(int)v16 + 14] )
  {
LABEL_24:
    if ( !*((_BYTE *)v15 + 500 * v16 + 6595) )
      goto LABEL_7;
  }
LABEL_17:
  v36 = HIWORD(v14);
  if ( sub_345A800((__int64)v15, a2, *(_QWORD **)(a1 + 8), 0, a3, v12, v14) )
  {
    v31 = *(_QWORD *)(a1 + 8);
    HIWORD(v19) = v36;
    v41 = *(_QWORD *)(a2 + 80);
    if ( v41 )
    {
      sub_3813810(&v41);
      HIWORD(v19) = v36;
    }
    LOWORD(v19) = v13;
    v26 = 215;
    v24 = v37;
    v27 = v31;
    v42 = *(_DWORD *)(a2 + 72);
    v25 = v19;
    goto LABEL_10;
  }
LABEL_7:
  v18 = sub_383B380(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v20 = *(_QWORD *)(a2 + 80);
  v21 = *(_QWORD *)(a1 + 8);
  v23 = (unsigned __int16 *)(*((_QWORD *)v18 + 6) + 16LL * v22);
  v24 = *((_QWORD *)v23 + 1);
  v25 = *v23;
  v41 = v20;
  if ( v20 )
  {
    v34 = v25;
    v38 = v24;
    sub_B96E90((__int64)&v41, v20, 1);
    v25 = v34;
    v24 = v38;
  }
  v26 = 189;
  v27 = v21;
  v42 = *(_DWORD *)(a2 + 72);
LABEL_10:
  v28 = sub_33FAF80(v27, v26, (__int64)&v41, v25, v24, v19, a3);
  if ( v41 )
    sub_B91220((__int64)&v41, v41);
  return v28;
}
