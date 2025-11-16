// Function: sub_3842EC0
// Address: 0x3842ec0
//
__int64 __fastcall sub_3842EC0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // r9
  __int64 v6; // rdx
  __int16 *v7; // rax
  unsigned __int16 v8; // si
  __int64 v9; // r8
  __int64 (__fastcall *v10)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v11; // r15
  unsigned int v12; // ecx
  int *v13; // r14
  char v14; // di
  __int64 v15; // r8
  int v16; // edx
  unsigned int v17; // r10d
  __int64 v18; // rsi
  int v19; // r9d
  __int128 v20; // rax
  __int64 v21; // r14
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rsi
  int v26; // esi
  int v27; // r11d
  __int64 v28; // rsi
  unsigned int v29; // [rsp+8h] [rbp-68h]
  __int64 v30; // [rsp+10h] [rbp-60h] BYREF
  int v31; // [rsp+18h] [rbp-58h]
  int v32; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int16 v33; // [rsp+28h] [rbp-48h]
  __int64 v34; // [rsp+30h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 80);
  v30 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v30, v4, 1);
  v5 = *(_QWORD *)a1;
  v6 = *(_QWORD *)(a1 + 8);
  v31 = *(_DWORD *)(a2 + 72);
  v7 = *(__int16 **)(a2 + 48);
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  v10 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v5 + 592LL);
  if ( v10 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v32, v5, *(_QWORD *)(v6 + 64), v8, v9);
    v11 = v34;
    v12 = v33;
  }
  else
  {
    v12 = v10(v5, *(_QWORD *)(v6 + 64), v8, v9);
    v11 = v24;
  }
  v29 = v12;
  v32 = sub_375D5B0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v13 = sub_3805BC0(a1 + 712, &v32);
  sub_37593F0(a1, v13);
  v14 = *(_BYTE *)(a1 + 512) & 1;
  if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
  {
    v15 = a1 + 520;
    v16 = 7;
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 528);
    v15 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v23 )
      goto LABEL_15;
    v16 = v23 - 1;
  }
  v17 = v16 & (37 * *v13);
  v18 = v15 + 24LL * v17;
  v19 = *(_DWORD *)v18;
  if ( *v13 == *(_DWORD *)v18 )
    goto LABEL_8;
  v26 = 1;
  while ( v19 != -1 )
  {
    v27 = v26 + 1;
    v28 = v16 & (v17 + v26);
    v17 = v28;
    v18 = v15 + 24 * v28;
    v19 = *(_DWORD *)v18;
    if ( *v13 == *(_DWORD *)v18 )
      goto LABEL_8;
    v26 = v27;
  }
  if ( v14 )
  {
    v25 = 192;
    goto LABEL_16;
  }
  v23 = *(unsigned int *)(a1 + 528);
LABEL_15:
  v25 = 24 * v23;
LABEL_16:
  v18 = v15 + v25;
LABEL_8:
  *(_QWORD *)&v20 = *(_QWORD *)(v18 + 8);
  *((_QWORD *)&v20 + 1) = *(unsigned int *)(v18 + 16);
  v21 = sub_340F900(
          *(_QWORD **)(a1 + 8),
          *(_DWORD *)(a2 + 24),
          (__int64)&v30,
          v29,
          v11,
          *(unsigned int *)(a2 + 24),
          v20,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  return v21;
}
