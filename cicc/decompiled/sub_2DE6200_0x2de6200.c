// Function: sub_2DE6200
// Address: 0x2de6200
//
__int64 __fastcall sub_2DE6200(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 (*v5)(); // rax
  __int64 v7; // r14
  __int64 v8; // r13
  __int64 v9; // r15
  char v10; // al
  __int64 v11; // r8
  __int64 (*v12)(); // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  int v15; // r11d
  unsigned int i; // eax
  __int64 v17; // rdi
  unsigned int v18; // eax
  unsigned __int64 *v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  _QWORD *v29; // rbx
  _QWORD *v30; // r15
  void (__fastcall *v31)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v32; // rax
  __int64 v34; // [rsp+10h] [rbp-350h] BYREF
  _QWORD *v35; // [rsp+18h] [rbp-348h]
  int v36; // [rsp+20h] [rbp-340h]
  int v37; // [rsp+24h] [rbp-33Ch]
  int v38; // [rsp+28h] [rbp-338h]
  char v39; // [rsp+2Ch] [rbp-334h]
  _QWORD v40[2]; // [rsp+30h] [rbp-330h] BYREF
  __int64 v41; // [rsp+40h] [rbp-320h] BYREF
  _BYTE *v42; // [rsp+48h] [rbp-318h]
  __int64 v43; // [rsp+50h] [rbp-310h]
  int v44; // [rsp+58h] [rbp-308h]
  char v45; // [rsp+5Ch] [rbp-304h]
  _BYTE v46[16]; // [rsp+60h] [rbp-300h] BYREF
  unsigned __int64 v47[2]; // [rsp+70h] [rbp-2F0h] BYREF
  _BYTE v48[512]; // [rsp+80h] [rbp-2E0h] BYREF
  __int64 v49; // [rsp+280h] [rbp-E0h]
  __int64 v50; // [rsp+288h] [rbp-D8h]
  unsigned __int64 *v51; // [rsp+290h] [rbp-D0h]
  __int64 v52; // [rsp+298h] [rbp-C8h]
  char v53; // [rsp+2A0h] [rbp-C0h]
  __int64 v54; // [rsp+2A8h] [rbp-B8h]
  char *v55; // [rsp+2B0h] [rbp-B0h]
  __int64 v56; // [rsp+2B8h] [rbp-A8h]
  int v57; // [rsp+2C0h] [rbp-A0h]
  char v58; // [rsp+2C4h] [rbp-9Ch]
  char v59; // [rsp+2C8h] [rbp-98h] BYREF
  __int16 v60; // [rsp+308h] [rbp-58h]
  _QWORD *v61; // [rsp+310h] [rbp-50h]
  _QWORD *v62; // [rsp+318h] [rbp-48h]
  __int64 v63; // [rsp+320h] [rbp-40h]

  v5 = *(__int64 (**)())(*(_QWORD *)*a2 + 16LL);
  if ( v5 == sub_23CE270 )
    BUG();
  v7 = a1 + 32;
  v8 = a1 + 80;
  v9 = ((__int64 (__fastcall *)(_QWORD, __int64))v5)(*a2, a3);
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 320LL))(v9);
  v11 = a3;
  if ( !v10 )
  {
    *(_QWORD *)(a1 + 8) = v7;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v8;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v12 = *(__int64 (**)())(*(_QWORD *)v9 + 144LL);
  if ( v12 == sub_2C8F680 )
  {
    v13 = *(unsigned int *)(a4 + 88);
    v14 = *(_QWORD *)(a4 + 72);
    if ( !(_DWORD)v13 )
      goto LABEL_38;
  }
  else
  {
    ((void (__fastcall *)(__int64))v12)(v9);
    v13 = *(unsigned int *)(a4 + 88);
    v11 = a3;
    v14 = *(_QWORD *)(a4 + 72);
    if ( !(_DWORD)v13 )
      goto LABEL_38;
  }
  v15 = 1;
  for ( i = (v13 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)))); ; i = (v13 - 1) & v18 )
  {
    v17 = v14 + 24LL * i;
    if ( *(_UNKNOWN **)v17 == &unk_4F81450 && v11 == *(_QWORD *)(v17 + 8) )
      break;
    if ( *(_QWORD *)v17 == -4096 && *(_QWORD *)(v17 + 8) == -4096 )
      goto LABEL_38;
    v18 = v15 + i;
    ++v15;
  }
  if ( v17 != v14 + 24 * v13 )
  {
    v20 = *(unsigned __int64 **)(*(_QWORD *)(v17 + 16) + 24LL);
    if ( v20 )
      ++v20;
    goto LABEL_15;
  }
LABEL_38:
  v20 = 0;
LABEL_15:
  v51 = v20;
  v47[0] = (unsigned __int64)v48;
  v47[1] = 0x1000000000LL;
  v55 = &v59;
  if ( v20 )
    v20 = v47;
  v60 = 0;
  v49 = 0;
  v50 = 0;
  v52 = 0;
  v53 = 1;
  v54 = 0;
  v56 = 8;
  v57 = 0;
  v58 = 1;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  if ( (unsigned __int8)sub_2DE4C60(v11, (__int64)v20) )
  {
    v35 = v40;
    v40[0] = &unk_4F81450;
    v36 = 2;
    v38 = 0;
    v39 = 1;
    v41 = 0;
    v42 = v46;
    v43 = 2;
    v44 = 0;
    v45 = 1;
    v37 = 1;
    v34 = 1;
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v40, (__int64)&v34);
    v20 = (unsigned __int64 *)(a1 + 80);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v46, (__int64)&v41);
    if ( !v45 )
      _libc_free((unsigned __int64)v42);
    if ( !v39 )
      _libc_free((unsigned __int64)v35);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v7;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v8;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  sub_FFCE90((__int64)v47, (__int64)v20, v21, v22, v23, v24);
  sub_FFD870((__int64)v47, (__int64)v20, v25, v26, v27, v28);
  sub_FFBC40((__int64)v47, (__int64)v20);
  v29 = v62;
  v30 = v61;
  if ( v62 != v61 )
  {
    do
    {
      v31 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v30[7];
      *v30 = &unk_49E5048;
      if ( v31 )
        v31(v30 + 5, v30 + 5, 3);
      *v30 = &unk_49DB368;
      v32 = v30[3];
      if ( v32 != 0 && v32 != -4096 && v32 != -8192 )
        sub_BD60C0(v30 + 1);
      v30 += 9;
    }
    while ( v29 != v30 );
    v30 = v61;
  }
  if ( v30 )
    j_j___libc_free_0((unsigned __int64)v30);
  if ( !v58 )
    _libc_free((unsigned __int64)v55);
  if ( (_BYTE *)v47[0] != v48 )
    _libc_free(v47[0]);
  return a1;
}
