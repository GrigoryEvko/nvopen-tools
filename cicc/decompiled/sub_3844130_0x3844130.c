// Function: sub_3844130
// Address: 0x3844130
//
unsigned __int8 *__fastcall sub_3844130(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v6; // r14
  char v7; // di
  __int64 v8; // r10
  int v9; // esi
  unsigned int v10; // r8d
  __int64 v11; // rax
  int v12; // r11d
  __int64 v13; // r10
  _QWORD *v14; // rbx
  __int64 v15; // rsi
  __int64 v16; // r11
  __int64 v17; // r14
  unsigned __int16 v18; // ax
  __int64 v19; // r14
  __int64 v20; // r8
  __int64 v21; // rcx
  __int64 v22; // rsi
  unsigned __int8 *v23; // rax
  __int64 v24; // rsi
  unsigned __int8 *v25; // r12
  __int64 v27; // rsi
  __int64 v28; // rcx
  unsigned int v29; // esi
  unsigned __int8 *v30; // rax
  __int64 v31; // rax
  int v32; // eax
  int v33; // ecx
  __int128 v34; // [rsp-40h] [rbp-B0h]
  __int128 v35; // [rsp-30h] [rbp-A0h]
  __int128 v36; // [rsp-20h] [rbp-90h]
  __int128 v37; // [rsp-10h] [rbp-80h]
  __int64 v38; // [rsp+0h] [rbp-70h]
  __int64 v39; // [rsp+0h] [rbp-70h]
  __int64 v40; // [rsp+8h] [rbp-68h]
  int *v41; // [rsp+10h] [rbp-60h]
  __int64 v42; // [rsp+10h] [rbp-60h]
  __int64 v43; // [rsp+10h] [rbp-60h]
  __int64 v44; // [rsp+18h] [rbp-58h]
  __int64 v45; // [rsp+20h] [rbp-50h]
  __int64 v46; // [rsp+28h] [rbp-48h]
  __int64 v47; // [rsp+30h] [rbp-40h] BYREF
  int v48; // [rsp+38h] [rbp-38h]

  v46 = sub_37AE0F0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v5 = v4;
  v6 = (unsigned int)v4;
  LODWORD(v47) = sub_375D5B0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v41 = sub_3805BC0(a1 + 712, (int *)&v47);
  sub_37593F0(a1, v41);
  v7 = *(_BYTE *)(a1 + 512) & 1;
  if ( v7 )
  {
    v8 = a1 + 520;
    v9 = 7;
  }
  else
  {
    v27 = *(unsigned int *)(a1 + 528);
    v8 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v27 )
      goto LABEL_17;
    v9 = v27 - 1;
  }
  v10 = v9 & (37 * *v41);
  v11 = v8 + 24LL * v10;
  v12 = *(_DWORD *)v11;
  if ( *v41 == *(_DWORD *)v11 )
    goto LABEL_4;
  v32 = 1;
  while ( v12 != -1 )
  {
    v33 = v32 + 1;
    v10 = v9 & (v32 + v10);
    v11 = v8 + 24LL * v10;
    v12 = *(_DWORD *)v11;
    if ( *v41 == *(_DWORD *)v11 )
      goto LABEL_4;
    v32 = v33;
  }
  if ( v7 )
  {
    v31 = 192;
    goto LABEL_18;
  }
  v27 = *(unsigned int *)(a1 + 528);
LABEL_17:
  v31 = 24 * v27;
LABEL_18:
  v11 = v8 + v31;
LABEL_4:
  v13 = *(_QWORD *)(v11 + 8);
  v14 = *(_QWORD **)(a1 + 8);
  v15 = *(_QWORD *)(a2 + 80);
  v16 = *(unsigned int *)(v11 + 16);
  v17 = *(_QWORD *)(v46 + 48) + 16 * v6;
  v18 = *(_WORD *)v17;
  v19 = *(_QWORD *)(v17 + 8);
  if ( *(_DWORD *)(a2 + 64) == 2 )
  {
    v47 = *(_QWORD *)(a2 + 80);
    v28 = v18;
    if ( v15 )
    {
      v39 = v18;
      v43 = v13;
      v44 = v16;
      sub_B96E90((__int64)&v47, v15, 1);
      v28 = v39;
      v13 = v43;
      v16 = v44;
    }
    *((_QWORD *)&v37 + 1) = v16;
    *(_QWORD *)&v37 = v13;
    v29 = *(_DWORD *)(a2 + 24);
    *((_QWORD *)&v36 + 1) = v5;
    *(_QWORD *)&v36 = v46;
    v48 = *(_DWORD *)(a2 + 72);
    v30 = sub_3406EB0(v14, v29, (__int64)&v47, v28, v19, (__int64)&v47, v36, v37);
    v24 = v47;
    v25 = v30;
    if ( v47 )
      goto LABEL_8;
  }
  else
  {
    v47 = *(_QWORD *)(a2 + 80);
    v20 = *(_QWORD *)(a2 + 40);
    v21 = v18;
    if ( v15 )
    {
      v45 = v18;
      v38 = v13;
      v40 = v16;
      v42 = *(_QWORD *)(a2 + 40);
      sub_B96E90((__int64)&v47, v15, 1);
      v21 = v45;
      v13 = v38;
      v16 = v40;
      v20 = v42;
    }
    v22 = *(unsigned int *)(a2 + 24);
    v48 = *(_DWORD *)(a2 + 72);
    *((_QWORD *)&v35 + 1) = v16;
    *(_QWORD *)&v35 = v13;
    *((_QWORD *)&v34 + 1) = v5;
    *(_QWORD *)&v34 = v46;
    v23 = sub_33FC130(
            v14,
            v22,
            (__int64)&v47,
            v21,
            v19,
            (__int64)&v47,
            v34,
            v35,
            *(_OWORD *)(v20 + 80),
            *(_OWORD *)(v20 + 120));
    v24 = v47;
    v25 = v23;
    if ( v47 )
LABEL_8:
      sub_B91220((__int64)&v47, v24);
  }
  return v25;
}
