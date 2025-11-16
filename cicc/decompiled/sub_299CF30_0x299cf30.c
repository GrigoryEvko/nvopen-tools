// Function: sub_299CF30
// Address: 0x299cf30
//
__int64 __fastcall sub_299CF30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  int v11; // r11d
  int v12; // edi
  unsigned int i; // eax
  __int64 v14; // r10
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r8
  int v19; // r11d
  unsigned int j; // eax
  __int64 v21; // r10
  unsigned int v22; // eax
  const char *v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  _QWORD *v32; // rbx
  _QWORD *v33; // r15
  void (__fastcall *v34)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v35; // rax
  __int64 v37; // [rsp+8h] [rbp-368h]
  __int64 v38; // [rsp+10h] [rbp-360h]
  __int64 v39; // [rsp+20h] [rbp-350h] BYREF
  _QWORD *v40; // [rsp+28h] [rbp-348h]
  int v41; // [rsp+30h] [rbp-340h]
  int v42; // [rsp+34h] [rbp-33Ch]
  int v43; // [rsp+38h] [rbp-338h]
  char v44; // [rsp+3Ch] [rbp-334h]
  _QWORD v45[2]; // [rsp+40h] [rbp-330h] BYREF
  __int64 v46; // [rsp+50h] [rbp-320h] BYREF
  _BYTE *v47; // [rsp+58h] [rbp-318h]
  __int64 v48; // [rsp+60h] [rbp-310h]
  int v49; // [rsp+68h] [rbp-308h]
  char v50; // [rsp+6Ch] [rbp-304h]
  _BYTE v51[16]; // [rsp+70h] [rbp-300h] BYREF
  unsigned __int64 v52[2]; // [rsp+80h] [rbp-2F0h] BYREF
  _BYTE v53[512]; // [rsp+90h] [rbp-2E0h] BYREF
  __int64 v54; // [rsp+290h] [rbp-E0h]
  __int64 v55; // [rsp+298h] [rbp-D8h]
  __int64 v56; // [rsp+2A0h] [rbp-D0h]
  __int64 v57; // [rsp+2A8h] [rbp-C8h]
  char v58; // [rsp+2B0h] [rbp-C0h]
  __int64 v59; // [rsp+2B8h] [rbp-B8h]
  char *v60; // [rsp+2C0h] [rbp-B0h]
  __int64 v61; // [rsp+2C8h] [rbp-A8h]
  int v62; // [rsp+2D0h] [rbp-A0h]
  char v63; // [rsp+2D4h] [rbp-9Ch]
  char v64; // [rsp+2D8h] [rbp-98h] BYREF
  __int16 v65; // [rsp+318h] [rbp-58h]
  _QWORD *v66; // [rsp+320h] [rbp-50h]
  _QWORD *v67; // [rsp+328h] [rbp-48h]
  __int64 v68; // [rsp+330h] [rbp-40h]

  v37 = sub_BC1CD0(a4, &unk_4F89C30, a3);
  v7 = sub_BC1CD0(a4, &unk_4F86540, a3);
  v8 = sub_BC1CD0(a4, &unk_4F8FAE8, a3);
  v9 = *(unsigned int *)(a4 + 88);
  v10 = *(_QWORD *)(a4 + 72);
  v38 = v8;
  if ( (_DWORD)v9 )
  {
    v11 = 1;
    v12 = v9 - 1;
    for ( i = (v9 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = v12 & v15 )
    {
      v14 = v10 + 24LL * i;
      if ( *(_UNKNOWN **)v14 == &unk_4F81450 && a3 == *(_QWORD *)(v14 + 8) )
        break;
      if ( *(_QWORD *)v14 == -4096 && *(_QWORD *)(v14 + 8) == -4096 )
        goto LABEL_7;
      v15 = v11 + i;
      ++v11;
    }
    v18 = v10 + 24 * v9;
    if ( v18 != v14 )
    {
      v16 = *(_QWORD *)(*(_QWORD *)(v14 + 16) + 24LL);
      if ( v16 )
        v16 += 8;
      goto LABEL_14;
    }
  }
  else
  {
LABEL_7:
    v14 = v10 + 24LL * (unsigned int)v9;
    if ( !(_DWORD)v9 )
    {
      v16 = 0;
LABEL_9:
      v17 = 0;
      goto LABEL_22;
    }
    v12 = v9 - 1;
  }
  v18 = v14;
  v16 = 0;
LABEL_14:
  v19 = 1;
  for ( j = v12
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F8FBC8 >> 9) ^ ((unsigned int)&unk_4F8FBC8 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = v12 & v22 )
  {
    v21 = v10 + 24LL * j;
    if ( *(_UNKNOWN **)v21 == &unk_4F8FBC8 && a3 == *(_QWORD *)(v21 + 8) )
      break;
    if ( *(_QWORD *)v21 == -4096 && *(_QWORD *)(v21 + 8) == -4096 )
      goto LABEL_9;
    v22 = v19 + j;
    ++v19;
  }
  if ( v21 == v18 )
    goto LABEL_9;
  v17 = *(_QWORD *)(*(_QWORD *)(v21 + 16) + 24LL);
  if ( v17 )
    v17 += 8;
LABEL_22:
  v57 = v17;
  v52[0] = (unsigned __int64)v53;
  v23 = "disable-tail-calls";
  v56 = v16;
  v60 = &v64;
  v52[1] = 0x1000000000LL;
  v54 = 0;
  v55 = 0;
  v58 = 0;
  v59 = 0;
  v61 = 8;
  v62 = 0;
  v63 = 1;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v39 = sub_B2D7E0(a3, "disable-tail-calls", 0x12u);
  if ( (unsigned __int8)sub_A72A30(&v39)
    || (v23 = (const char *)(v37 + 8),
        !(unsigned __int8)sub_299AC30(a3, v37 + 8, v7 + 8, (__int64 *)(v38 + 8), (__int64)v52)) )
  {
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  else
  {
    v40 = v45;
    v41 = 2;
    v43 = 0;
    v44 = 1;
    v46 = 0;
    v47 = v51;
    v48 = 2;
    v49 = 0;
    v50 = 1;
    v42 = 1;
    v45[0] = &unk_4F81450;
    v39 = 1;
    if ( &unk_4F81450 != (_UNKNOWN *)&qword_4F82400 && &unk_4F81450 != &unk_4F8FBC8 )
    {
      v42 = 2;
      v39 = 2;
      v45[1] = &unk_4F8FBC8;
    }
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v45, (__int64)&v39);
    v23 = (const char *)(a1 + 80);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v51, (__int64)&v46);
    if ( !v50 )
      _libc_free((unsigned __int64)v47);
    if ( !v44 )
      _libc_free((unsigned __int64)v40);
  }
  sub_FFCE90((__int64)v52, (__int64)v23, v24, v25, v26, v27);
  sub_FFD870((__int64)v52, (__int64)v23, v28, v29, v30, v31);
  sub_FFBC40((__int64)v52, (__int64)v23);
  v32 = v67;
  v33 = v66;
  if ( v67 != v66 )
  {
    do
    {
      v34 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v33[7];
      *v33 = &unk_49E5048;
      if ( v34 )
        v34(v33 + 5, v33 + 5, 3);
      *v33 = &unk_49DB368;
      v35 = v33[3];
      if ( v35 != -4096 && v35 != 0 && v35 != -8192 )
        sub_BD60C0(v33 + 1);
      v33 += 9;
    }
    while ( v32 != v33 );
    v33 = v66;
  }
  if ( v33 )
    j_j___libc_free_0((unsigned __int64)v33);
  if ( !v63 )
    _libc_free((unsigned __int64)v60);
  if ( (_BYTE *)v52[0] != v53 )
    _libc_free(v52[0]);
  return a1;
}
