// Function: sub_3843BE0
// Address: 0x3843be0
//
__int64 __fastcall sub_3843BE0(__int64 a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, int a7)
{
  __int16 *v9; // rax
  __int64 v10; // r10
  unsigned __int16 v11; // si
  __int64 v12; // r8
  __int64 (__fastcall *v13)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // r8
  unsigned int v18; // eax
  __int64 v19; // rdx
  unsigned int v20; // r9d
  __int64 v21; // r8
  __int64 v22; // rsi
  int *v23; // r14
  char v24; // si
  __int64 v25; // rdi
  int v26; // ecx
  int v27; // edx
  unsigned int v28; // r11d
  __int64 v29; // rax
  int v30; // r10d
  __int64 v31; // r14
  __int64 v32; // r15
  __int128 v33; // rax
  __int64 v34; // r9
  __int64 v35; // r14
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rax
  int v41; // eax
  int v42; // r14d
  __int64 v43; // rax
  __int128 v44; // [rsp-40h] [rbp-C0h]
  unsigned int v45; // [rsp+0h] [rbp-80h]
  unsigned int v46; // [rsp+0h] [rbp-80h]
  __int64 v47; // [rsp+8h] [rbp-78h]
  __int64 v48; // [rsp+8h] [rbp-78h]
  __int16 v49; // [rsp+Ah] [rbp-76h]
  int v50; // [rsp+1Ch] [rbp-64h] BYREF
  unsigned int v51; // [rsp+20h] [rbp-60h] BYREF
  __int64 v52; // [rsp+28h] [rbp-58h]
  __int64 v53; // [rsp+30h] [rbp-50h] BYREF
  int v54; // [rsp+38h] [rbp-48h]
  __int64 v55; // [rsp+40h] [rbp-40h]

  v9 = *(__int16 **)(a2 + 48);
  v10 = *(_QWORD *)a1;
  v49 = HIWORD(a7);
  v11 = *v9;
  v12 = *((_QWORD *)v9 + 1);
  v13 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(**(_QWORD **)a1 + 592LL);
  if ( v13 == sub_2D56A50 )
  {
    v14 = v11;
    v15 = *(_QWORD *)a1;
    sub_2FE6CC0((__int64)&v53, v10, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL), v14, v12);
    LOWORD(v18) = v54;
    v19 = v55;
    HIWORD(v20) = v49;
    LOWORD(v51) = v54;
    v52 = v55;
  }
  else
  {
    v39 = v11;
    v15 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL);
    v18 = v13(v10, v15, v39, v12);
    HIWORD(v20) = v49;
    v51 = v18;
    v52 = v19;
  }
  if ( (_WORD)v18 )
  {
    v21 = 0;
    LOWORD(v18) = word_4456580[(unsigned __int16)v18 - 1];
  }
  else
  {
    v18 = sub_3009970((__int64)&v51, v15, v19, v16, v17);
    HIWORD(v20) = HIWORD(v18);
    v21 = v38;
  }
  v22 = *(_QWORD *)(a2 + 80);
  LOWORD(v20) = v18;
  v53 = v22;
  if ( v22 )
  {
    v45 = v20;
    v47 = v21;
    sub_B96E90((__int64)&v53, v22, 1);
    v20 = v45;
    v21 = v47;
  }
  v46 = v20;
  v48 = v21;
  v54 = *(_DWORD *)(a2 + 72);
  v50 = sub_375D5B0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v23 = sub_3805BC0(a1 + 712, &v50);
  sub_37593F0(a1, v23);
  v24 = *(_BYTE *)(a1 + 512) & 1;
  if ( v24 )
  {
    v25 = a1 + 520;
    v26 = 7;
  }
  else
  {
    v37 = *(unsigned int *)(a1 + 528);
    v25 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v37 )
      goto LABEL_18;
    v26 = v37 - 1;
  }
  v27 = *v23;
  v28 = v26 & (37 * *v23);
  v29 = v25 + 24LL * v28;
  v30 = *(_DWORD *)v29;
  if ( *v23 == *(_DWORD *)v29 )
    goto LABEL_10;
  v41 = 1;
  while ( v30 != -1 )
  {
    v42 = v41 + 1;
    v43 = v26 & (v28 + v41);
    v28 = v43;
    v29 = v25 + 24 * v43;
    v30 = *(_DWORD *)v29;
    if ( v27 == *(_DWORD *)v29 )
      goto LABEL_10;
    v41 = v42;
  }
  if ( v24 )
  {
    v40 = 192;
    goto LABEL_19;
  }
  v37 = *(unsigned int *)(a1 + 528);
LABEL_18:
  v40 = 24 * v37;
LABEL_19:
  v29 = v25 + v40;
LABEL_10:
  v31 = *(_QWORD *)(v29 + 8);
  v32 = *(unsigned int *)(v29 + 16);
  *(_QWORD *)&v33 = sub_33FAF80(*(_QWORD *)(a1 + 8), 215, (__int64)&v53, v46, v48, v46, a3);
  *((_QWORD *)&v44 + 1) = v32;
  *(_QWORD *)&v44 = v31;
  v35 = sub_340F900(
          *(_QWORD **)(a1 + 8),
          0x9Du,
          (__int64)&v53,
          v51,
          v52,
          v34,
          v44,
          v33,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  if ( v53 )
    sub_B91220((__int64)&v53, v53);
  return v35;
}
