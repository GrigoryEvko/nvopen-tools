// Function: sub_383A2C0
// Address: 0x383a2c0
//
__int64 *__fastcall sub_383A2C0(__int64 a1, __int64 a2)
{
  int *v4; // r14
  char v5; // si
  __int64 v6; // r8
  int v7; // ecx
  unsigned int v8; // edi
  __int64 v9; // rax
  int v10; // r9d
  int v11; // esi
  __int64 v12; // r14
  __int64 *v13; // r11
  __int128 *v14; // r12
  const __m128i *v15; // r9
  __int64 v16; // r8
  __int64 v17; // r15
  __int64 *v18; // r10
  __int64 v19; // rax
  unsigned __int16 v20; // cx
  __int64 *v21; // r14
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // eax
  int v26; // r10d
  __int128 v27; // [rsp-20h] [rbp-90h]
  unsigned __int16 v28; // [rsp+8h] [rbp-68h]
  const __m128i *v29; // [rsp+10h] [rbp-60h]
  __int64 *v30; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+20h] [rbp-50h]
  __int64 *v32; // [rsp+28h] [rbp-48h]
  __int64 v33; // [rsp+30h] [rbp-40h] BYREF
  int v34; // [rsp+38h] [rbp-38h]

  LODWORD(v33) = sub_375D5B0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v4 = sub_3805BC0(a1 + 712, (int *)&v33);
  sub_37593F0(a1, v4);
  v5 = *(_BYTE *)(a1 + 512) & 1;
  if ( v5 )
  {
    v6 = a1 + 520;
    v7 = 7;
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 528);
    v6 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v23 )
      goto LABEL_14;
    v7 = v23 - 1;
  }
  v8 = v7 & (37 * *v4);
  v9 = v6 + 24LL * v8;
  v10 = *(_DWORD *)v9;
  if ( *v4 == *(_DWORD *)v9 )
    goto LABEL_4;
  v25 = 1;
  while ( v10 != -1 )
  {
    v26 = v25 + 1;
    v8 = v7 & (v25 + v8);
    v9 = v6 + 24LL * v8;
    v10 = *(_DWORD *)v9;
    if ( *v4 == *(_DWORD *)v9 )
      goto LABEL_4;
    v25 = v26;
  }
  if ( v5 )
  {
    v24 = 192;
    goto LABEL_15;
  }
  v23 = *(unsigned int *)(a1 + 528);
LABEL_14:
  v24 = 24 * v23;
LABEL_15:
  v9 = v6 + v24;
LABEL_4:
  v11 = *(_DWORD *)(a2 + 24);
  v12 = *(_QWORD *)(v9 + 8);
  v13 = *(__int64 **)(a1 + 8);
  v14 = *(__int128 **)(a2 + 40);
  v15 = *(const __m128i **)(a2 + 112);
  v16 = *(_QWORD *)(a2 + 104);
  v17 = *(unsigned int *)(v9 + 16);
  v18 = (__int64 *)v14 + 5;
  if ( v11 == 339 )
    v18 = (__int64 *)(v14 + 5);
  v19 = *(_QWORD *)(a2 + 80);
  v20 = *(_WORD *)(a2 + 96);
  v33 = v19;
  if ( v19 )
  {
    v28 = v20;
    v29 = v15;
    v30 = v18;
    v31 = v16;
    v32 = v13;
    sub_B96E90((__int64)&v33, v19, 1);
    v11 = *(_DWORD *)(a2 + 24);
    v20 = v28;
    v15 = v29;
    v18 = v30;
    v16 = v31;
    v13 = v32;
  }
  v34 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v27 + 1) = v17;
  *(_QWORD *)&v27 = v12;
  v21 = sub_33F34C0(v13, v11, (__int64)&v33, v20, v16, v15, *v14, v27, *v18, v18[1]);
  if ( v33 )
    sub_B91220((__int64)&v33, v33);
  return v21;
}
