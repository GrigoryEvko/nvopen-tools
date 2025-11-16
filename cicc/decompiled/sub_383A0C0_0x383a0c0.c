// Function: sub_383A0C0
// Address: 0x383a0c0
//
__m128i *__fastcall sub_383A0C0(__int64 a1, __int64 a2)
{
  int *v4; // r14
  char v5; // si
  __int64 v6; // r8
  int v7; // ecx
  unsigned int v8; // edi
  __int64 v9; // rax
  int v10; // r9d
  unsigned __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rsi
  __int64 *v14; // r13
  const __m128i *v15; // r11
  char v16; // al
  __int64 v17; // r14
  __int64 v18; // r15
  unsigned __int64 *v19; // rax
  __int128 *v20; // r10
  __int128 *v21; // rcx
  __m128i *v22; // r14
  __int64 v24; // rax
  __int64 v25; // rax
  int v26; // eax
  int v27; // r10d
  unsigned __int64 v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+10h] [rbp-70h]
  const __m128i *v30; // [rsp+18h] [rbp-68h]
  __int128 *v31; // [rsp+20h] [rbp-60h]
  __int128 *v32; // [rsp+28h] [rbp-58h]
  char v33; // [rsp+34h] [rbp-4Ch]
  __int128 *v34; // [rsp+38h] [rbp-48h]
  __int64 v35; // [rsp+40h] [rbp-40h] BYREF
  int v36; // [rsp+48h] [rbp-38h]

  LODWORD(v35) = sub_375D5B0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v4 = sub_3805BC0(a1 + 712, (int *)&v35);
  sub_37593F0(a1, v4);
  v5 = *(_BYTE *)(a1 + 512) & 1;
  if ( v5 )
  {
    v6 = a1 + 520;
    v7 = 7;
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 528);
    v6 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v24 )
      goto LABEL_12;
    v7 = v24 - 1;
  }
  v8 = v7 & (37 * *v4);
  v9 = v6 + 24LL * v8;
  v10 = *(_DWORD *)v9;
  if ( *v4 == *(_DWORD *)v9 )
    goto LABEL_4;
  v26 = 1;
  while ( v10 != -1 )
  {
    v27 = v26 + 1;
    v8 = v7 & (v26 + v8);
    v9 = v6 + 24LL * v8;
    v10 = *(_DWORD *)v9;
    if ( *v4 == *(_DWORD *)v9 )
      goto LABEL_4;
    v26 = v27;
  }
  if ( v5 )
  {
    v25 = 192;
    goto LABEL_13;
  }
  v24 = *(unsigned int *)(a1 + 528);
LABEL_12:
  v25 = 24 * v24;
LABEL_13:
  v9 = v6 + v25;
LABEL_4:
  v11 = *(_QWORD *)(v9 + 8);
  v12 = *(unsigned int *)(v9 + 16);
  v13 = *(_QWORD *)(a2 + 80);
  v14 = *(__int64 **)(a1 + 8);
  v15 = *(const __m128i **)(a2 + 112);
  v16 = *(_BYTE *)(a2 + 33) >> 3;
  v17 = *(unsigned __int16 *)(a2 + 96);
  v18 = *(_QWORD *)(a2 + 104);
  v35 = v13;
  v33 = v16 & 1;
  v19 = *(unsigned __int64 **)(a2 + 40);
  v20 = (__int128 *)(v19 + 25);
  v34 = (__int128 *)(v19 + 10);
  v21 = (__int128 *)(v19 + 20);
  if ( v13 )
  {
    v28 = v11;
    v29 = v12;
    v30 = v15;
    v31 = (__int128 *)(v19 + 25);
    v32 = (__int128 *)(v19 + 20);
    sub_B96E90((__int64)&v35, v13, 1);
    v19 = *(unsigned __int64 **)(a2 + 40);
    v11 = v28;
    v12 = v29;
    v15 = v30;
    v20 = v31;
    v21 = v32;
  }
  v36 = *(_DWORD *)(a2 + 72);
  v22 = sub_33F5840(v14, *v19, v19[1], (__int64)&v35, v11, v12, *v34, *v21, *v20, v17, v18, v15, v33);
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
  return v22;
}
