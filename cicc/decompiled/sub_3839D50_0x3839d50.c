// Function: sub_3839D50
// Address: 0x3839d50
//
unsigned __int8 *__fastcall sub_3839D50(__int64 a1, __int64 a2)
{
  int *v4; // r12
  char v5; // si
  __int64 v6; // r8
  int v7; // ecx
  unsigned int v8; // edi
  __int64 v9; // rax
  int v10; // r9d
  _QWORD *v11; // rbx
  __int64 v12; // r9
  __int64 v13; // rsi
  __int64 v14; // r12
  __int64 v15; // rdx
  unsigned __int16 *v16; // rax
  __int64 v17; // r8
  __int64 v18; // rcx
  __int64 v19; // r13
  unsigned __int8 *v20; // r12
  __int64 v22; // rax
  __int64 v23; // rax
  int v24; // eax
  int v25; // r10d
  __int128 v26; // [rsp-20h] [rbp-80h]
  __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+18h] [rbp-48h]
  __int64 v30; // [rsp+20h] [rbp-40h] BYREF
  int v31; // [rsp+28h] [rbp-38h]

  LODWORD(v30) = sub_375D5B0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v4 = sub_3805BC0(a1 + 712, (int *)&v30);
  sub_37593F0(a1, v4);
  v5 = *(_BYTE *)(a1 + 512) & 1;
  if ( v5 )
  {
    v6 = a1 + 520;
    v7 = 7;
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 528);
    v6 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v22 )
      goto LABEL_12;
    v7 = v22 - 1;
  }
  v8 = v7 & (37 * *v4);
  v9 = v6 + 24LL * v8;
  v10 = *(_DWORD *)v9;
  if ( *v4 == *(_DWORD *)v9 )
    goto LABEL_4;
  v24 = 1;
  while ( v10 != -1 )
  {
    v25 = v24 + 1;
    v8 = v7 & (v24 + v8);
    v9 = v6 + 24LL * v8;
    v10 = *(_DWORD *)v9;
    if ( *v4 == *(_DWORD *)v9 )
      goto LABEL_4;
    v24 = v25;
  }
  if ( v5 )
  {
    v23 = 192;
    goto LABEL_13;
  }
  v22 = *(unsigned int *)(a1 + 528);
LABEL_12:
  v23 = 24 * v22;
LABEL_13:
  v9 = v6 + v23;
LABEL_4:
  v11 = *(_QWORD **)(a1 + 8);
  v12 = *(_QWORD *)(a2 + 40);
  v13 = *(_QWORD *)(a2 + 80);
  v14 = *(_QWORD *)(v9 + 8);
  v15 = *(unsigned int *)(v9 + 16);
  v16 = (unsigned __int16 *)(*(_QWORD *)(v14 + 48) + 16 * v15);
  v17 = *((_QWORD *)v16 + 1);
  v18 = *v16;
  v19 = v15;
  v30 = v13;
  if ( v13 )
  {
    v27 = v18;
    v28 = v12;
    v29 = v17;
    sub_B96E90((__int64)&v30, v13, 1);
    v18 = v27;
    v12 = v28;
    v17 = v29;
  }
  v31 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v26 + 1) = v19;
  *(_QWORD *)&v26 = v14;
  v20 = sub_3406EB0(v11, 0xDEu, (__int64)&v30, v18, v17, v12, v26, *(_OWORD *)(v12 + 40));
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  return v20;
}
