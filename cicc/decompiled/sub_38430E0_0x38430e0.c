// Function: sub_38430E0
// Address: 0x38430e0
//
__int64 *__fastcall sub_38430E0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  unsigned __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // rsi
  unsigned __int16 *v8; // rax
  unsigned int v9; // ecx
  int *v10; // rbx
  __int64 v11; // r8
  int v12; // esi
  unsigned int v13; // edi
  __int64 v14; // rax
  int v15; // r9d
  _QWORD *v16; // r15
  __int64 v17; // rsi
  __int64 v18; // rbx
  __int128 v19; // rax
  unsigned __int8 *v20; // r14
  __int64 v21; // rdx
  __int64 v22; // r15
  __int64 v24; // rsi
  __int64 v25; // rax
  int v26; // eax
  int v27; // r10d
  __int128 v28; // [rsp-20h] [rbp-90h]
  unsigned int v29; // [rsp+0h] [rbp-70h]
  unsigned int v30; // [rsp+0h] [rbp-70h]
  __int64 v31; // [rsp+0h] [rbp-70h]
  unsigned __int64 v32; // [rsp+10h] [rbp-60h]
  _QWORD *v33; // [rsp+18h] [rbp-58h]
  int v34; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v35; // [rsp+30h] [rbp-40h] BYREF
  int v36; // [rsp+38h] [rbp-38h]

  v33 = *(_QWORD **)(a1 + 8);
  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(v4 + 40);
  v6 = *(_QWORD *)(v4 + 48);
  v7 = *(_QWORD *)(v5 + 80);
  v8 = (unsigned __int16 *)(*(_QWORD *)(v5 + 48) + 16LL * *(unsigned int *)(v4 + 48));
  v32 = *((_QWORD *)v8 + 1);
  v9 = *v8;
  v35 = v7;
  if ( v7 )
  {
    v29 = v9;
    sub_B96E90((__int64)&v35, v7, 1);
    v9 = v29;
  }
  v30 = v9;
  v36 = *(_DWORD *)(v5 + 72);
  v34 = sub_375D5B0(a1, v5, v6);
  v10 = sub_3805BC0(a1 + 712, &v34);
  sub_37593F0(a1, v10);
  if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
  {
    v11 = a1 + 520;
    v12 = 7;
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 528);
    v11 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v24 )
      goto LABEL_12;
    v12 = v24 - 1;
  }
  v13 = v12 & (37 * *v10);
  v14 = v11 + 24LL * v13;
  v15 = *(_DWORD *)v14;
  if ( *v10 == *(_DWORD *)v14 )
    goto LABEL_6;
  v26 = 1;
  while ( v15 != -1 )
  {
    v27 = v26 + 1;
    v13 = v12 & (v26 + v13);
    v14 = v11 + 24LL * v13;
    v15 = *(_DWORD *)v14;
    if ( *v10 == *(_DWORD *)v14 )
      goto LABEL_6;
    v26 = v27;
  }
  if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
  {
    v25 = 192;
    goto LABEL_13;
  }
  v24 = *(unsigned int *)(a1 + 528);
LABEL_12:
  v25 = 24 * v24;
LABEL_13:
  v14 = v11 + v25;
LABEL_6:
  v16 = *(_QWORD **)(a1 + 8);
  v17 = v30;
  v18 = *(unsigned int *)(v14 + 16);
  v31 = *(_QWORD *)(v14 + 8);
  *(_QWORD *)&v19 = sub_33F7D60(v16, v17, v32);
  *((_QWORD *)&v28 + 1) = v18 | v6 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v28 = v31;
  v20 = sub_3406EB0(
          v16,
          0xDEu,
          (__int64)&v35,
          *(unsigned __int16 *)(*(_QWORD *)(v31 + 48) + 16 * v18),
          *(_QWORD *)(*(_QWORD *)(v31 + 48) + 16 * v18 + 8),
          v31,
          v28,
          v19);
  v22 = v21;
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
  return sub_33EC010(
           v33,
           (__int64 *)a2,
           **(_QWORD **)(a2 + 40),
           *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
           (__int64)v20,
           v22);
}
