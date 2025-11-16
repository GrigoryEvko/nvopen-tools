// Function: sub_373D2D0
// Address: 0x373d2d0
//
unsigned __int64 __fastcall sub_373D2D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r13
  __int64 v7; // rax
  unsigned int v8; // esi
  __int64 v9; // r14
  __int64 v10; // r9
  int v11; // r11d
  __int64 *v12; // rcx
  unsigned int v13; // r8d
  __int64 *v14; // rax
  __int64 v15; // rdi
  unsigned __int64 v16; // rcx
  __int64 v17; // rax
  unsigned __int64 v18; // r14
  _QWORD *v19; // rdx
  __int64 v20; // rcx
  unsigned int *v21; // r8
  __int64 v22; // r9
  __int64 v23; // rbx
  unsigned __int8 v24; // al
  _BYTE **v25; // rdx
  _BYTE *v26; // rsi
  unsigned __int8 v27; // al
  _BYTE **v28; // rsi
  unsigned int v29; // eax
  __int64 v30; // r8
  unsigned __int8 v31; // al
  __int64 v32; // rdx
  int v34; // eax
  int v35; // esi
  __int64 v36; // rdx
  unsigned int v37; // eax
  int v38; // edi
  __int64 v39; // r8
  int v40; // r10d
  __int64 *v41; // r9
  unsigned __int8 v42; // al
  __int64 v43; // rbx
  __int64 v44; // r8
  int v45; // eax
  int v46; // eax
  int v47; // eax
  __int64 v48; // r8
  int v49; // r10d
  unsigned int v50; // edx
  __int64 v51; // rsi
  unsigned __int64 v52; // [rsp+8h] [rbp-48h]
  __int64 v53; // [rsp+8h] [rbp-48h]
  unsigned int v54; // [rsp+8h] [rbp-48h]
  int v55; // [rsp+1Ch] [rbp-34h]

  v6 = sub_AE7A60(*(_BYTE **)(a2 + 8));
  if ( !sub_3734FE0(a1) || (v9 = a1 + 672, (unsigned __int8)sub_321F6A0(*(_QWORD *)(a1 + 208), a2)) )
  {
    v7 = *(_QWORD *)(a1 + 216);
    v8 = *(_DWORD *)(v7 + 424);
    v9 = v7 + 400;
    if ( !v8 )
    {
LABEL_28:
      ++*(_QWORD *)v9;
      goto LABEL_29;
    }
  }
  else
  {
    v8 = *(_DWORD *)(a1 + 696);
    if ( !v8 )
      goto LABEL_28;
  }
  v10 = *(_QWORD *)(v9 + 8);
  v11 = 1;
  v12 = 0;
  v13 = (v8 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v14 = (__int64 *)(v10 + 16LL * v13);
  v15 = *v14;
  if ( v6 == *v14 )
  {
LABEL_4:
    v16 = v14[1];
    goto LABEL_5;
  }
  while ( v15 != -4096 )
  {
    if ( v15 == -8192 && !v12 )
      v12 = v14;
    v13 = (v8 - 1) & (v11 + v13);
    v14 = (__int64 *)(v10 + 16LL * v13);
    v15 = *v14;
    if ( v6 == *v14 )
      goto LABEL_4;
    ++v11;
  }
  if ( !v12 )
    v12 = v14;
  v45 = *(_DWORD *)(v9 + 16);
  ++*(_QWORD *)v9;
  v38 = v45 + 1;
  if ( 4 * (v45 + 1) >= 3 * v8 )
  {
LABEL_29:
    sub_373B830(v9, 2 * v8);
    v34 = *(_DWORD *)(v9 + 24);
    if ( v34 )
    {
      v35 = v34 - 1;
      v36 = *(_QWORD *)(v9 + 8);
      v37 = (v34 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v38 = *(_DWORD *)(v9 + 16) + 1;
      v12 = (__int64 *)(v36 + 16LL * v37);
      v39 = *v12;
      if ( v6 != *v12 )
      {
        v40 = 1;
        v41 = 0;
        while ( v39 != -4096 )
        {
          if ( v39 == -8192 && !v41 )
            v41 = v12;
          v37 = v35 & (v40 + v37);
          v12 = (__int64 *)(v36 + 16LL * v37);
          v39 = *v12;
          if ( v6 == *v12 )
            goto LABEL_51;
          ++v40;
        }
LABEL_33:
        if ( v41 )
          v12 = v41;
        goto LABEL_51;
      }
      goto LABEL_51;
    }
LABEL_69:
    ++*(_DWORD *)(v9 + 16);
    BUG();
  }
  if ( v8 - *(_DWORD *)(v9 + 20) - v38 > v8 >> 3 )
    goto LABEL_51;
  v54 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
  sub_373B830(v9, v8);
  v46 = *(_DWORD *)(v9 + 24);
  if ( !v46 )
    goto LABEL_69;
  v47 = v46 - 1;
  v48 = *(_QWORD *)(v9 + 8);
  v49 = 1;
  v41 = 0;
  v50 = v47 & v54;
  v38 = *(_DWORD *)(v9 + 16) + 1;
  v12 = (__int64 *)(v48 + 16LL * (v47 & v54));
  v51 = *v12;
  if ( v6 != *v12 )
  {
    while ( v51 != -4096 )
    {
      if ( v51 == -8192 && !v41 )
        v41 = v12;
      v50 = v47 & (v49 + v50);
      v12 = (__int64 *)(v48 + 16LL * v50);
      v51 = *v12;
      if ( v6 == *v12 )
        goto LABEL_51;
      ++v49;
    }
    goto LABEL_33;
  }
LABEL_51:
  *(_DWORD *)(v9 + 16) = v38;
  if ( *v12 != -4096 )
    --*(_DWORD *)(v9 + 20);
  *v12 = v6;
  v12[1] = 0;
  v16 = 0;
LABEL_5:
  v52 = v16;
  v17 = sub_A777F0(0x30u, (__int64 *)(a1 + 88));
  v18 = v17;
  if ( v17 )
  {
    *(_QWORD *)(v17 + 8) = 0;
    *(_QWORD *)(v17 + 16) = 0;
    *(_QWORD *)v17 = v17 | 4;
    *(_DWORD *)(v17 + 24) = -1;
    *(_WORD *)(v17 + 28) = 29;
    *(_BYTE *)(v17 + 30) = 0;
    *(_QWORD *)(v17 + 32) = 0;
  }
  else
  {
    v17 = 0;
  }
  *(_QWORD *)(v18 + 40) = a3 & 0xFFFFFFFFFFFFFFFBLL;
  v19 = *(_QWORD **)(a3 + 32);
  if ( v19 )
  {
    *(_QWORD *)v18 = *v19;
    **(_QWORD **)(a3 + 32) = v17 & 0xFFFFFFFFFFFFFFFBLL;
  }
  *(_QWORD *)(a3 + 32) = v17;
  sub_32494F0((__int64 *)a1, v18, 49, v52);
  sub_373CB80((__int64 *)a1, v18, a2 + 80, v20, v21, v22);
  v23 = *(_QWORD *)(a2 + 16);
  v53 = v23 - 16;
  v24 = *(_BYTE *)(v23 - 16);
  if ( (v24 & 2) != 0 )
    v25 = *(_BYTE ***)(v23 - 32);
  else
    v25 = (_BYTE **)(v53 - 8LL * ((v24 >> 2) & 0xF));
  v26 = *v25;
  if ( **v25 != 16 )
  {
    v27 = *(v26 - 16);
    if ( (v27 & 2) != 0 )
      v28 = (_BYTE **)*((_QWORD *)v26 - 4);
    else
      v28 = (_BYTE **)&v26[-8 * ((v27 >> 2) & 0xF) - 16];
    v26 = *v28;
  }
  v29 = sub_373B2C0(a1, (__int64)v26);
  BYTE2(v55) = 0;
  sub_3249A20((__int64 *)a1, (unsigned __int64 **)(v18 + 8), 88, v55, v29);
  BYTE2(v55) = 0;
  sub_3249A20((__int64 *)a1, (unsigned __int64 **)(v18 + 8), 89, v55, *(unsigned int *)(v23 + 4));
  v30 = *(unsigned __int16 *)(v23 + 2);
  if ( (_WORD)v30 )
  {
    BYTE2(v55) = 0;
    sub_3249A20((__int64 *)a1, (unsigned __int64 **)(v18 + 8), 87, v55, v30);
  }
  v31 = *(_BYTE *)(v23 - 16);
  if ( (v31 & 2) != 0 )
    v32 = *(_QWORD *)(v23 - 32);
  else
    v32 = v53 - 8LL * ((v31 >> 2) & 0xF);
  if ( **(_BYTE **)v32 == 20
    && *(_DWORD *)(*(_QWORD *)v32 + 4LL)
    && (unsigned __int16)sub_3220AA0(*(_QWORD *)(a1 + 208)) > 3u )
  {
    v42 = *(_BYTE *)(v23 - 16);
    if ( (v42 & 2) != 0 )
      v43 = *(_QWORD *)(v23 - 32);
    else
      v43 = v53 - 8LL * ((v42 >> 2) & 0xF);
    v44 = 0;
    if ( **(_BYTE **)v43 == 20 )
      v44 = *(unsigned int *)(*(_QWORD *)v43 + 4LL);
    BYTE2(v55) = 0;
    sub_3249A20((__int64 *)a1, (unsigned __int64 **)(v18 + 8), 8502, v55, v44);
  }
  sub_32379A0(*(_QWORD *)(a1 + 208), a1, *(_DWORD *)(*(_QWORD *)(a1 + 80) + 36LL), v6, v18);
  return v18;
}
