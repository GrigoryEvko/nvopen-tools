// Function: sub_3843520
// Address: 0x3843520
//
unsigned __int8 *__fastcall sub_3843520(__int64 a1, __int64 a2, __m128i a3)
{
  int *v5; // r12
  char v6; // si
  __int64 v7; // r8
  int v8; // ecx
  unsigned int v9; // edi
  int *v10; // rax
  int v11; // r10d
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rdi
  unsigned __int8 *v16; // rax
  _QWORD *v17; // rbx
  unsigned int v18; // edx
  unsigned __int8 *v19; // r14
  __int64 v20; // r12
  unsigned __int64 v21; // r15
  unsigned __int16 *v22; // rax
  __int128 v23; // rax
  unsigned __int8 *v24; // r12
  __int64 v26; // rax
  __int64 v27; // rax
  int v28; // eax
  int v29; // r11d
  __int128 v30; // [rsp-30h] [rbp-80h]
  unsigned __int8 *v31; // [rsp+8h] [rbp-48h]
  __int64 v32; // [rsp+10h] [rbp-40h] BYREF
  int v33; // [rsp+18h] [rbp-38h]

  LODWORD(v32) = sub_375D5B0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v5 = sub_3805BC0(a1 + 712, (int *)&v32);
  sub_37593F0(a1, v5);
  v6 = *(_BYTE *)(a1 + 512) & 1;
  if ( v6 )
  {
    v7 = a1 + 520;
    v8 = 7;
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 528);
    v7 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v26 )
      goto LABEL_12;
    v8 = v26 - 1;
  }
  v9 = v8 & (37 * *v5);
  v10 = (int *)(v7 + 24LL * v9);
  v11 = *v10;
  if ( *v5 == *v10 )
    goto LABEL_4;
  v28 = 1;
  while ( v11 != -1 )
  {
    v29 = v28 + 1;
    v9 = v8 & (v28 + v9);
    v10 = (int *)(v7 + 24LL * v9);
    v11 = *v10;
    if ( *v5 == *v10 )
      goto LABEL_4;
    v28 = v29;
  }
  if ( v6 )
  {
    v27 = 192;
    goto LABEL_13;
  }
  v26 = *(unsigned int *)(a1 + 528);
LABEL_12:
  v27 = 24 * v26;
LABEL_13:
  v10 = (int *)(v7 + v27);
LABEL_4:
  v12 = *(_QWORD *)(a2 + 80);
  v13 = (unsigned int)v10[4];
  v32 = v12;
  v14 = v13;
  if ( v12 )
    sub_B96E90((__int64)&v32, v12, 1);
  v15 = *(_QWORD *)(a1 + 8);
  v33 = *(_DWORD *)(a2 + 72);
  v16 = sub_33FAF80(
          v15,
          215,
          (__int64)&v32,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          (unsigned int)&v32,
          a3);
  v17 = *(_QWORD **)(a1 + 8);
  v31 = v16;
  v19 = v16;
  v20 = 16LL * v18;
  v21 = v18 | v14 & 0xFFFFFFFF00000000LL;
  v22 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL));
  *(_QWORD *)&v23 = sub_33F7D60(v17, *v22, *((_QWORD *)v22 + 1));
  *((_QWORD *)&v30 + 1) = v21;
  *(_QWORD *)&v30 = v19;
  v24 = sub_3406EB0(
          v17,
          0xDEu,
          (__int64)&v32,
          *(unsigned __int16 *)(*((_QWORD *)v31 + 6) + v20),
          *(_QWORD *)(*((_QWORD *)v31 + 6) + v20 + 8),
          (__int64)&v32,
          v30,
          v23);
  if ( v32 )
    sub_B91220((__int64)&v32, v32);
  return v24;
}
