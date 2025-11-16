// Function: sub_383B180
// Address: 0x383b180
//
unsigned __int8 *__fastcall sub_383B180(__int64 a1, __int64 a2)
{
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // r13
  char v7; // si
  __int64 v8; // rdi
  int v9; // ecx
  unsigned int v10; // r9d
  __int64 v11; // rax
  int v12; // r8d
  __int64 v13; // r10
  __int64 v14; // rsi
  _QWORD *v15; // r9
  __int128 *v16; // rbx
  __int64 v17; // r11
  __int64 v18; // r8
  __int64 v19; // rcx
  unsigned __int8 *v20; // r12
  __int64 v22; // rdx
  __int64 v23; // rax
  int v24; // eax
  int v25; // r10d
  __int64 v26; // rax
  __int128 v27; // [rsp-30h] [rbp-A0h]
  __int128 v28; // [rsp-20h] [rbp-90h]
  __int64 v29; // [rsp+8h] [rbp-68h]
  __int64 v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+20h] [rbp-50h]
  int *v33; // [rsp+28h] [rbp-48h]
  _QWORD *v34; // [rsp+28h] [rbp-48h]
  __int64 v35; // [rsp+30h] [rbp-40h] BYREF
  int v36; // [rsp+38h] [rbp-38h]

  v4 = sub_37AE0F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
  v6 = v5;
  LODWORD(v35) = sub_375D5B0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 120LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 128LL));
  v33 = sub_3805BC0(a1 + 712, (int *)&v35);
  sub_37593F0(a1, v33);
  v7 = *(_BYTE *)(a1 + 512) & 1;
  if ( v7 )
  {
    v8 = a1 + 520;
    v9 = 7;
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 528);
    v8 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v22 )
      goto LABEL_12;
    v9 = v22 - 1;
  }
  v10 = v9 & (37 * *v33);
  v11 = v8 + 24LL * v10;
  v12 = *(_DWORD *)v11;
  if ( *v33 == *(_DWORD *)v11 )
    goto LABEL_4;
  v24 = 1;
  while ( v12 != -1 )
  {
    v25 = v24 + 1;
    v26 = v9 & (v10 + v24);
    v10 = v26;
    v11 = v8 + 24 * v26;
    v12 = *(_DWORD *)v11;
    if ( *v33 == *(_DWORD *)v11 )
      goto LABEL_4;
    v24 = v25;
  }
  if ( v7 )
  {
    v23 = 192;
    goto LABEL_13;
  }
  v22 = *(unsigned int *)(a1 + 528);
LABEL_12:
  v23 = 24 * v22;
LABEL_13:
  v11 = v8 + v23;
LABEL_4:
  v13 = *(_QWORD *)(v11 + 8);
  v14 = *(_QWORD *)(a2 + 80);
  v15 = *(_QWORD **)(a1 + 8);
  v16 = *(__int128 **)(a2 + 40);
  v17 = *(unsigned int *)(v11 + 16);
  v18 = *(_QWORD *)(*(_QWORD *)(v4 + 48) + 16LL * (unsigned int)v6 + 8);
  v19 = *(unsigned __int16 *)(*(_QWORD *)(v4 + 48) + 16LL * (unsigned int)v6);
  v35 = v14;
  if ( v14 )
  {
    v29 = v19;
    v30 = v13;
    v31 = v17;
    v32 = v18;
    v34 = v15;
    sub_B96E90((__int64)&v35, v14, 1);
    v19 = v29;
    v13 = v30;
    v17 = v31;
    v18 = v32;
    v15 = v34;
  }
  v36 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v28 + 1) = v17;
  *(_QWORD *)&v28 = v13;
  *((_QWORD *)&v27 + 1) = v6;
  *(_QWORD *)&v27 = v4;
  v20 = sub_33FC1D0(
          v15,
          207,
          (__int64)&v35,
          v19,
          v18,
          (__int64)v15,
          *v16,
          *(__int128 *)((char *)v16 + 40),
          v27,
          v28,
          v16[10]);
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
  return v20;
}
