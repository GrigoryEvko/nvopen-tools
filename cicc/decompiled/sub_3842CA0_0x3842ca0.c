// Function: sub_3842CA0
// Address: 0x3842ca0
//
unsigned __int8 *__fastcall sub_3842CA0(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rsi
  int *v6; // r14
  char v7; // di
  __int64 v8; // r9
  int v9; // esi
  int v10; // edx
  unsigned int v11; // r11d
  __int64 v12; // rax
  int v13; // r10d
  __int64 v14; // r15
  __int64 v15; // r14
  unsigned int v16; // edx
  __int64 v17; // rcx
  __int64 v18; // rax
  unsigned __int8 *v19; // r14
  __int64 v21; // rax
  __int64 v22; // rax
  int v23; // eax
  int v24; // r14d
  __int64 v25; // rax
  __int128 v26; // [rsp-30h] [rbp-A0h]
  __int64 v27; // [rsp+0h] [rbp-70h]
  unsigned int v28; // [rsp+8h] [rbp-68h]
  int v29; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v30; // [rsp+30h] [rbp-40h] BYREF
  int v31; // [rsp+38h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v30 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v30, v5, 1);
  v31 = *(_DWORD *)(a2 + 72);
  v27 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v28 = **(unsigned __int16 **)(a2 + 48);
  v29 = sub_375D5B0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = sub_3805BC0(a1 + 712, &v29);
  sub_37593F0(a1, v6);
  v7 = *(_BYTE *)(a1 + 512) & 1;
  if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
  {
    v8 = a1 + 520;
    v9 = 7;
  }
  else
  {
    v21 = *(unsigned int *)(a1 + 528);
    v8 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v21 )
      goto LABEL_12;
    v9 = v21 - 1;
  }
  v10 = *v6;
  v11 = v9 & (37 * *v6);
  v12 = v8 + 24LL * v11;
  v13 = *(_DWORD *)v12;
  if ( *v6 == *(_DWORD *)v12 )
    goto LABEL_6;
  v23 = 1;
  while ( v13 != -1 )
  {
    v24 = v23 + 1;
    v25 = v9 & (v11 + v23);
    v11 = v25;
    v12 = v8 + 24 * v25;
    v13 = *(_DWORD *)v12;
    if ( v10 == *(_DWORD *)v12 )
      goto LABEL_6;
    v23 = v24;
  }
  if ( v7 )
  {
    v22 = 192;
    goto LABEL_13;
  }
  v21 = *(unsigned int *)(a1 + 528);
LABEL_12:
  v22 = 24 * v21;
LABEL_13:
  v12 = v8 + v22;
LABEL_6:
  v14 = *(unsigned int *)(v12 + 16);
  *((_QWORD *)&v26 + 1) = v14;
  *(_QWORD *)&v26 = *(_QWORD *)(v12 + 8);
  v15 = sub_340F900(
          *(_QWORD **)(a1 + 8),
          0x1CBu,
          (__int64)&v30,
          v28,
          v27,
          0xFFFFFFFF00000000LL,
          v26,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  v17 = *(_QWORD *)(a2 + 40);
  v18 = *(_QWORD *)(*(_QWORD *)v17 + 48LL) + 16LL * *(unsigned int *)(v17 + 8);
  v19 = sub_3400810(
          *(_QWORD **)(a1 + 8),
          v15,
          v16 | v14 & 0xFFFFFFFF00000000LL,
          *(_QWORD *)(v17 + 40),
          *(_QWORD *)(v17 + 48),
          (__int64)&v30,
          a3,
          *(_OWORD *)(v17 + 80),
          *(_WORD *)v18,
          *(_QWORD *)(v18 + 8));
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  return v19;
}
