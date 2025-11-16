// Function: sub_3843990
// Address: 0x3843990
//
unsigned __int8 *__fastcall sub_3843990(__int64 a1, __int64 a2, __m128i a3)
{
  unsigned int *v5; // rax
  unsigned __int64 v6; // r8
  __int64 v7; // r14
  __int64 v8; // rsi
  unsigned __int16 *v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // r15d
  char v12; // si
  __int64 v13; // rdi
  int v14; // ecx
  unsigned int v15; // r8d
  __int64 v16; // rax
  int v17; // r9d
  unsigned __int8 *v18; // rax
  __int64 v19; // rdx
  _QWORD *v20; // r9
  __int64 v21; // rsi
  __int64 v22; // r10
  unsigned __int8 *v23; // r14
  __int64 v24; // r8
  __int64 v25; // rcx
  __int64 v26; // r15
  unsigned __int8 *v27; // r14
  __int64 v29; // rdx
  __int64 v30; // rax
  int v31; // eax
  int v32; // r11d
  __int128 v33; // [rsp-20h] [rbp-90h]
  __int64 v34; // [rsp+0h] [rbp-70h]
  __int64 v35; // [rsp+8h] [rbp-68h]
  unsigned __int64 v36; // [rsp+10h] [rbp-60h]
  int *v37; // [rsp+10h] [rbp-60h]
  __int64 v38; // [rsp+10h] [rbp-60h]
  __int64 v39; // [rsp+10h] [rbp-60h]
  __int64 v40; // [rsp+18h] [rbp-58h]
  unsigned __int8 *v41; // [rsp+18h] [rbp-58h]
  _QWORD *v42; // [rsp+18h] [rbp-58h]
  int v43; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v44; // [rsp+30h] [rbp-40h] BYREF
  int v45; // [rsp+38h] [rbp-38h]

  v5 = *(unsigned int **)(a2 + 40);
  v6 = *(_QWORD *)v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = *(_QWORD *)(*(_QWORD *)v5 + 80LL);
  v9 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v5 + 48LL) + 16LL * v5[2]);
  v10 = *((_QWORD *)v9 + 1);
  v11 = *v9;
  v44 = v8;
  v40 = v10;
  if ( v8 )
  {
    v36 = v6;
    sub_B96E90((__int64)&v44, v8, 1);
    v6 = v36;
  }
  v45 = *(_DWORD *)(v6 + 72);
  v43 = sub_375D5B0(a1, v6, v7);
  v37 = sub_3805BC0(a1 + 712, &v43);
  sub_37593F0(a1, v37);
  v12 = *(_BYTE *)(a1 + 512) & 1;
  if ( v12 )
  {
    v13 = a1 + 520;
    v14 = 7;
  }
  else
  {
    v29 = *(unsigned int *)(a1 + 528);
    v13 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v29 )
      goto LABEL_16;
    v14 = v29 - 1;
  }
  v15 = v14 & (37 * *v37);
  v16 = v13 + 24LL * v15;
  v17 = *(_DWORD *)v16;
  if ( *v37 == *(_DWORD *)v16 )
    goto LABEL_6;
  v31 = 1;
  while ( v17 != -1 )
  {
    v32 = v31 + 1;
    v15 = v14 & (v31 + v15);
    v16 = v13 + 24LL * v15;
    v17 = *(_DWORD *)v16;
    if ( *v37 == *(_DWORD *)v16 )
      goto LABEL_6;
    v31 = v32;
  }
  if ( v12 )
  {
    v30 = 192;
    goto LABEL_17;
  }
  v29 = *(unsigned int *)(a1 + 528);
LABEL_16:
  v30 = 24 * v29;
LABEL_17:
  v16 = v13 + v30;
LABEL_6:
  v18 = sub_34070B0(
          *(_QWORD **)(a1 + 8),
          *(_QWORD *)(v16 + 8),
          *(unsigned int *)(v16 + 16) | v7 & 0xFFFFFFFF00000000LL,
          (__int64)&v44,
          v11,
          v40,
          a3);
  if ( v44 )
  {
    v38 = v19;
    v41 = v18;
    sub_B91220((__int64)&v44, v44);
    v19 = v38;
    v18 = v41;
  }
  v20 = *(_QWORD **)(a1 + 8);
  v21 = *(_QWORD *)(a2 + 80);
  v22 = *(_QWORD *)(a2 + 40);
  v23 = v18;
  v24 = *(_QWORD *)(*((_QWORD *)v18 + 6) + 16LL * (unsigned int)v19 + 8);
  v25 = *(unsigned __int16 *)(*((_QWORD *)v18 + 6) + 16LL * (unsigned int)v19);
  v26 = v19;
  v44 = v21;
  if ( v21 )
  {
    v34 = v25;
    v35 = v22;
    v39 = v24;
    v42 = v20;
    sub_B96E90((__int64)&v44, v21, 1);
    v25 = v34;
    v22 = v35;
    v24 = v39;
    v20 = v42;
  }
  v45 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v33 + 1) = v26;
  *(_QWORD *)&v33 = v23;
  v27 = sub_3406EB0(v20, 4u, (__int64)&v44, v25, v24, (__int64)v20, v33, *(_OWORD *)(v22 + 40));
  if ( v44 )
    sub_B91220((__int64)&v44, v44);
  return v27;
}
