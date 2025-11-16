// Function: sub_3843750
// Address: 0x3843750
//
__int64 __fastcall sub_3843750(__int64 a1, __int64 a2, __m128i a3)
{
  int *v5; // r14
  char v6; // si
  __int64 v7; // r8
  int v8; // ecx
  unsigned int v9; // edi
  __int64 v10; // rax
  int v11; // r9d
  __int64 v12; // r14
  _QWORD *v13; // r9
  __int64 v14; // rsi
  __int64 v15; // r15
  unsigned __int16 *v16; // rax
  __int64 v17; // r8
  unsigned int v18; // ebx
  unsigned __int8 *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // r14
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // rcx
  unsigned int v26; // ebx
  __int64 v27; // rax
  __int64 v28; // rax
  int v29; // eax
  int v30; // r10d
  __int128 v31; // [rsp-30h] [rbp-90h]
  _QWORD *v32; // [rsp+8h] [rbp-58h]
  _QWORD *v33; // [rsp+10h] [rbp-50h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+18h] [rbp-48h]
  __int64 v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+20h] [rbp-40h] BYREF
  int v38; // [rsp+28h] [rbp-38h]

  LODWORD(v37) = sub_375D5B0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v5 = sub_3805BC0(a1 + 712, (int *)&v37);
  sub_37593F0(a1, v5);
  v6 = *(_BYTE *)(a1 + 512) & 1;
  if ( v6 )
  {
    v7 = a1 + 520;
    v8 = 7;
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 528);
    v7 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v23 )
      goto LABEL_17;
    v8 = v23 - 1;
  }
  v9 = v8 & (37 * *v5);
  v10 = v7 + 24LL * v9;
  v11 = *(_DWORD *)v10;
  if ( *v5 == *(_DWORD *)v10 )
    goto LABEL_4;
  v29 = 1;
  while ( v11 != -1 )
  {
    v30 = v29 + 1;
    v9 = v8 & (v29 + v9);
    v10 = v7 + 24LL * v9;
    v11 = *(_DWORD *)v10;
    if ( *v5 == *(_DWORD *)v10 )
      goto LABEL_4;
    v29 = v30;
  }
  if ( v6 )
  {
    v28 = 192;
    goto LABEL_18;
  }
  v23 = *(unsigned int *)(a1 + 528);
LABEL_17:
  v28 = 24 * v23;
LABEL_18:
  v10 = v7 + v28;
LABEL_4:
  v12 = *(_QWORD *)(v10 + 8);
  v13 = *(_QWORD **)(a1 + 8);
  v14 = *(_QWORD *)(a2 + 80);
  v15 = *(unsigned int *)(v10 + 16);
  v16 = *(unsigned __int16 **)(a2 + 48);
  if ( *(_DWORD *)(a2 + 24) == 458 )
  {
    v24 = *((_QWORD *)v16 + 1);
    v25 = *(_QWORD *)(a2 + 40);
    v26 = *v16;
    v37 = *(_QWORD *)(a2 + 80);
    if ( v14 )
    {
      v32 = v13;
      v34 = v25;
      v36 = v24;
      sub_B96E90((__int64)&v37, v14, 1);
      v13 = v32;
      v25 = v34;
      v24 = v36;
    }
    v38 = *(_DWORD *)(a2 + 72);
    *((_QWORD *)&v31 + 1) = v15;
    *(_QWORD *)&v31 = v12;
    v27 = sub_340F900(
            v13,
            0x1CAu,
            (__int64)&v37,
            v26,
            v24,
            (__int64)v13,
            v31,
            *(_OWORD *)(v25 + 40),
            *(_OWORD *)(v25 + 80));
    v20 = v37;
    v21 = v27;
    if ( v37 )
      goto LABEL_8;
  }
  else
  {
    v17 = *((_QWORD *)v16 + 1);
    v18 = *v16;
    v37 = *(_QWORD *)(a2 + 80);
    if ( v14 )
    {
      v33 = v13;
      v35 = v17;
      sub_B96E90((__int64)&v37, v14, 1);
      v13 = v33;
      v17 = v35;
    }
    v38 = *(_DWORD *)(a2 + 72);
    v19 = sub_33FAF80((__int64)v13, 216, (__int64)&v37, v18, v17, (_DWORD)v13, a3);
    v20 = v37;
    v21 = (__int64)v19;
    if ( v37 )
LABEL_8:
      sub_B91220((__int64)&v37, v20);
  }
  return v21;
}
