// Function: sub_2A3A9B0
// Address: 0x2a3a9b0
//
__int64 __fastcall sub_2A3A9B0(__int64 *a1, int a2)
{
  __int64 v3; // r10
  __int64 v4; // r11
  __int64 v5; // r13
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // rsi
  int v9; // edi
  __int64 v10; // rax
  unsigned int v11; // edi
  _QWORD *v12; // rax
  __int64 v13; // r11
  __int64 v14; // r12
  __int64 v15; // r15
  __int64 v16; // rsi
  __int64 v17; // r13
  __int64 v18; // r13
  __int64 v19; // r14
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 v22; // r14
  __int64 v23; // rax
  _BYTE *v24; // rax
  __int64 v25; // rdi
  __int64 (__fastcall *v26)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v27; // rax
  _BYTE **v28; // rcx
  __int64 v29; // r13
  __int64 v31; // r10
  __int64 v32; // r12
  __int64 v33; // rbx
  __int64 v34; // r12
  __int64 v35; // rdx
  unsigned int v36; // esi
  __int64 v37; // rdx
  int v38; // eax
  char v39; // cl
  __int64 v40; // rdx
  int v41; // r13d
  __int64 *v42; // rax
  int v43; // eax
  int v44; // [rsp+Ch] [rbp-F4h]
  __int64 v45; // [rsp+10h] [rbp-F0h]
  __int64 v46; // [rsp+18h] [rbp-E8h]
  __int64 v47; // [rsp+18h] [rbp-E8h]
  __int64 v48; // [rsp+20h] [rbp-E0h]
  __int64 v49; // [rsp+20h] [rbp-E0h]
  int v50; // [rsp+28h] [rbp-D8h]
  unsigned int v51; // [rsp+2Ch] [rbp-D4h]
  _BYTE *v52; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v53; // [rsp+38h] [rbp-C8h] BYREF
  char v54[32]; // [rsp+40h] [rbp-C0h] BYREF
  __int16 v55; // [rsp+60h] [rbp-A0h]
  char v56[32]; // [rsp+70h] [rbp-90h] BYREF
  __int16 v57; // [rsp+90h] [rbp-70h]
  unsigned __int64 v58; // [rsp+A0h] [rbp-60h] BYREF
  unsigned int v59; // [rsp+A8h] [rbp-58h]
  unsigned __int64 v60; // [rsp+B0h] [rbp-50h]
  unsigned int v61; // [rsp+B8h] [rbp-48h]
  __int16 v62; // [rsp+C0h] [rbp-40h]

  v3 = sub_B6E160(*(__int64 **)(*(_QWORD *)(a1[6] + 72) + 40LL), 0x160u, 0, 0);
  v57 = 257;
  v55 = 257;
  v4 = 0;
  v51 = 8 * a2;
  if ( v3 )
    v4 = *(_QWORD *)(v3 + 24);
  v5 = a1[15];
  v6 = a1[14];
  v62 = 257;
  v7 = v6 + 56 * v5;
  if ( v6 == v7 )
  {
    v44 = 1;
    v11 = 1;
  }
  else
  {
    v8 = v6;
    v9 = 0;
    do
    {
      v10 = *(_QWORD *)(v8 + 40) - *(_QWORD *)(v8 + 32);
      v8 += 56;
      v9 += v10 >> 3;
    }
    while ( v7 != v8 );
    v11 = v9 + 1;
    v44 = v11 & 0x7FFFFFF;
  }
  v45 = v6;
  v46 = v4;
  LOBYTE(v50) = 16 * (_DWORD)v5 != 0;
  v48 = v3;
  v12 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v5) << 32) | v11);
  v13 = v46;
  v14 = (__int64)v12;
  if ( v12 )
  {
    v15 = v5;
    v16 = **(_QWORD **)(v46 + 16);
    v17 = (__int64)v12;
    v47 = v48;
    v49 = v13;
    sub_B44260((__int64)v12, v16, 56, v44 | (v50 << 28), 0, 0);
    *(_QWORD *)(v14 + 72) = 0;
    sub_B4A290(v14, v49, v47, 0, 0, (__int64)&v58, v45, v15);
  }
  else
  {
    v17 = 0;
  }
  if ( *((_BYTE *)a1 + 108) )
  {
    v42 = (__int64 *)sub_BD5C60(v17);
    *(_QWORD *)(v14 + 72) = sub_A7A090((__int64 *)(v14 + 72), v42, -1, 72);
  }
  if ( (unsigned __int8)sub_920620(v17) )
  {
    v40 = a1[12];
    v41 = *((_DWORD *)a1 + 26);
    if ( v40 )
      sub_B99FD0(v14, 3u, v40);
    sub_B45150(v14, v41);
  }
  (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v14,
    v54,
    a1[7],
    a1[8]);
  v18 = *a1;
  v19 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v19 )
  {
    do
    {
      v20 = *(_QWORD *)(v18 + 8);
      v21 = *(_DWORD *)v18;
      v18 += 16;
      sub_B99FD0(v14, v21, v20);
    }
    while ( v19 != v18 );
  }
  v22 = sub_BCB2B0((_QWORD *)a1[9]);
  v23 = sub_BCB2D0((_QWORD *)a1[9]);
  v24 = (_BYTE *)sub_ACD640(v23, v51, 0);
  v25 = a1[10];
  v52 = v24;
  v26 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v25 + 64LL);
  if ( v26 == sub_920540 )
  {
    if ( sub_BCEA30(v22) )
      goto LABEL_21;
    if ( *(_BYTE *)v14 > 0x15u )
      goto LABEL_21;
    v27 = sub_2A39930(&v52, (__int64)&v53);
    if ( v28 != v27 )
      goto LABEL_21;
    LOBYTE(v62) = 0;
    v29 = sub_AD9FD0(v22, (unsigned __int8 *)v14, (__int64 *)&v52, 1, 0, (__int64)&v58, 0);
    if ( (_BYTE)v62 )
    {
      LOBYTE(v62) = 0;
      if ( v61 > 0x40 && v60 )
        j_j___libc_free_0_0(v60);
      if ( v59 > 0x40 && v58 )
        j_j___libc_free_0_0(v58);
    }
  }
  else
  {
    v29 = v26(v25, v22, (_BYTE *)v14, &v52, 1, 0);
  }
  if ( v29 )
    return v29;
LABEL_21:
  v62 = 257;
  v29 = (__int64)sub_BD2C40(88, 2u);
  if ( !v29 )
    goto LABEL_24;
  v31 = *(_QWORD *)(v14 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17 > 1 )
  {
    v37 = *((_QWORD *)v52 + 1);
    v38 = *(unsigned __int8 *)(v37 + 8);
    if ( v38 == 17 )
    {
      v39 = 0;
    }
    else
    {
      v39 = 1;
      if ( v38 != 18 )
        goto LABEL_23;
    }
    v43 = *(_DWORD *)(v37 + 32);
    BYTE4(v53) = v39;
    LODWORD(v53) = v43;
    v31 = sub_BCE1B0((__int64 *)v31, v53);
  }
LABEL_23:
  sub_B44260(v29, v31, 34, 2u, 0, 0);
  *(_QWORD *)(v29 + 72) = v22;
  *(_QWORD *)(v29 + 80) = sub_B4DC50(v22, (__int64)&v52, 1);
  sub_B4D9A0(v29, v14, (__int64 *)&v52, 1, (__int64)&v58);
LABEL_24:
  (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v29,
    v56,
    a1[7],
    a1[8]);
  v32 = 16LL * *((unsigned int *)a1 + 2);
  v33 = *a1;
  v34 = v33 + v32;
  while ( v34 != v33 )
  {
    v35 = *(_QWORD *)(v33 + 8);
    v36 = *(_DWORD *)v33;
    v33 += 16;
    sub_B99FD0(v29, v36, v35);
  }
  return v29;
}
