// Function: sub_3842830
// Address: 0x3842830
//
__int64 *__fastcall sub_3842830(__int64 a1, unsigned __int64 a2)
{
  int *v4; // r14
  char v5; // si
  __int64 v6; // r8
  int v7; // ecx
  unsigned int v8; // edi
  __int64 v9; // rax
  int v10; // r9d
  __int64 v11; // r14
  __int128 *v12; // r10
  int v13; // esi
  __int64 *v14; // r11
  const __m128i *v15; // r9
  __int64 v16; // r8
  __int64 v17; // r15
  __int128 *v18; // rax
  unsigned __int16 v19; // cx
  __int64 v20; // rax
  __int64 *v21; // r14
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // eax
  int v26; // r10d
  unsigned __int16 v27; // [rsp+0h] [rbp-70h]
  __int128 *v28; // [rsp+8h] [rbp-68h]
  const __m128i *v29; // [rsp+10h] [rbp-60h]
  __int64 v30; // [rsp+18h] [rbp-58h]
  __int64 *v31; // [rsp+20h] [rbp-50h]
  __int128 *v32; // [rsp+28h] [rbp-48h]
  __int64 v33; // [rsp+30h] [rbp-40h] BYREF
  int v34; // [rsp+38h] [rbp-38h]

  LODWORD(v33) = sub_375D5B0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
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
  v11 = *(_QWORD *)(v9 + 8);
  v12 = *(__int128 **)(a2 + 40);
  v13 = *(_DWORD *)(a2 + 24);
  v14 = *(__int64 **)(a1 + 8);
  v15 = *(const __m128i **)(a2 + 112);
  v16 = *(_QWORD *)(a2 + 104);
  v17 = *(unsigned int *)(v9 + 16);
  v18 = v12 + 5;
  if ( v13 != 339 )
    v18 = (__int128 *)((char *)v12 + 40);
  v19 = *(_WORD *)(a2 + 96);
  v32 = v18;
  v20 = *(_QWORD *)(a2 + 80);
  v33 = v20;
  if ( v20 )
  {
    v27 = v19;
    v28 = v12;
    v29 = v15;
    v30 = v16;
    v31 = v14;
    sub_B96E90((__int64)&v33, v20, 1);
    v13 = *(_DWORD *)(a2 + 24);
    v19 = v27;
    v12 = v28;
    v15 = v29;
    v16 = v30;
    v14 = v31;
  }
  v34 = *(_DWORD *)(a2 + 72);
  v21 = sub_33F34C0(v14, v13, (__int64)&v33, v19, v16, v15, *v12, *v32, v11, v17);
  if ( v33 )
    sub_B91220((__int64)&v33, v33);
  sub_3760E70(a1, a2, 1, (unsigned __int64)v21, 1);
  return v21;
}
