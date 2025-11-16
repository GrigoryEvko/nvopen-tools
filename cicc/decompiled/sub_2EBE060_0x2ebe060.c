// Function: sub_2EBE060
// Address: 0x2ebe060
//
void __fastcall sub_2EBE060(__int64 *a1, __int64 a2)
{
  _QWORD *v4; // r14
  _QWORD *v5; // r13
  unsigned __int64 v6; // rsi
  _QWORD *v7; // rax
  _QWORD *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rcx
  _QWORD *v12; // r8
  __int64 v13; // rax
  _QWORD *v14; // rdi
  __int64 v15; // rdx
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r12
  int v21; // ecx
  unsigned __int64 v22; // rdx
  int v23; // ecx
  unsigned __int64 v24; // r8
  int v25; // ecx
  void *v26; // rax
  void *v27; // rcx
  unsigned __int64 v28; // rdi
  unsigned int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // rdi
  __int64 (*v32)(); // rdx
  __int64 v33; // r14

  *a1 = a2;
  a1[1] = 0;
  a1[2] = (__int64)(a1 + 5);
  a1[3] = 1;
  *((_DWORD *)a1 + 8) = 0;
  *((_BYTE *)a1 + 36) = 1;
  v4 = sub_C52410();
  v5 = v4 + 1;
  v6 = sub_C959E0();
  v7 = (_QWORD *)v4[2];
  if ( v7 )
  {
    v8 = v4 + 1;
    do
    {
      while ( 1 )
      {
        v9 = v7[2];
        v10 = v7[3];
        if ( v6 <= v7[4] )
          break;
        v7 = (_QWORD *)v7[3];
        if ( !v10 )
          goto LABEL_6;
      }
      v8 = v7;
      v7 = (_QWORD *)v7[2];
    }
    while ( v9 );
LABEL_6:
    if ( v5 != v8 && v6 >= v8[4] )
      v5 = v8;
  }
  if ( v5 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_38;
  v13 = v5[7];
  v12 = v5 + 6;
  if ( !v13 )
    goto LABEL_38;
  v6 = (unsigned int)dword_5020A08;
  v14 = v5 + 6;
  do
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(v13 + 16);
      v15 = *(_QWORD *)(v13 + 24);
      if ( *(_DWORD *)(v13 + 32) >= dword_5020A08 )
        break;
      v13 = *(_QWORD *)(v13 + 24);
      if ( !v15 )
        goto LABEL_15;
    }
    v14 = (_QWORD *)v13;
    v13 = *(_QWORD *)(v13 + 16);
  }
  while ( v11 );
LABEL_15:
  if ( v12 == v14 || dword_5020A08 < *((_DWORD *)v14 + 8) || (v16 = qword_5020A88, !*((_DWORD *)v14 + 9)) )
  {
LABEL_38:
    v31 = *(_QWORD *)(a2 + 16);
    v32 = *(__int64 (**)())(*(_QWORD *)v31 + 440LL);
    v16 = 0;
    if ( v32 != sub_2EBDF60 )
      v16 = ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64 (*)(), __int64, _QWORD *))v32)(
              v31,
              v6,
              v32,
              v11,
              v12);
  }
  *((_BYTE *)a1 + 48) = v16;
  a1[12] = (__int64)(a1 + 14);
  a1[14] = (__int64)(a1 + 16);
  a1[21] = 0x800000000LL;
  a1[23] = (__int64)(a1 + 26);
  a1[30] = (__int64)(a1 + 32);
  a1[33] = (__int64)(a1 + 35);
  a1[34] = 0x400000000LL;
  a1[40] = 0x600000000LL;
  a1[49] = 0x600000000LL;
  a1[48] = (__int64)(a1 + 50);
  a1[57] = (__int64)(a1 + 59);
  a1[7] = (__int64)(a1 + 9);
  a1[8] = 0;
  a1[9] = 0;
  a1[10] = 0;
  a1[13] = 0;
  a1[15] = 0;
  *((_BYTE *)a1 + 128) = 0;
  a1[19] = 0;
  a1[20] = 0;
  *((_BYTE *)a1 + 176) = 0;
  a1[24] = 0;
  a1[25] = 16;
  a1[31] = 0;
  *((_DWORD *)a1 + 64) = 0;
  a1[38] = 0;
  a1[39] = (__int64)(a1 + 41);
  *((_DWORD *)a1 + 94) = 0;
  *((_DWORD *)a1 + 112) = 0;
  a1[58] = 0;
  a1[59] = 0;
  a1[61] = 0;
  v17 = *a1;
  a1[62] = 0;
  a1[63] = 0;
  v20 = *(unsigned int *)((*(__int64 (__fastcall **)(_QWORD, unsigned __int64, __int64 *, __int64))(**(_QWORD **)(v17 + 16)
                                                                                                  + 200LL))(
                            *(_QWORD *)(v17 + 16),
                            v6,
                            a1 + 50,
                            v11)
                        + 16);
  if ( *((_DWORD *)a1 + 17) <= 0xFFu )
  {
    v6 = (unsigned __int64)(a1 + 9);
    sub_C8D5F0((__int64)(a1 + 7), a1 + 9, 0x100u, 0x10u, v18, v19);
  }
  v21 = a1[47] & 0x3F;
  if ( v21 )
  {
    v6 = *((unsigned int *)a1 + 80);
    *(_QWORD *)(a1[39] + 8 * v6 - 8) &= ~(-1LL << v21);
  }
  v22 = *((unsigned int *)a1 + 80);
  *((_DWORD *)a1 + 94) = v20;
  LOBYTE(v23) = v20;
  v24 = (unsigned int)(v20 + 63) >> 6;
  if ( v24 != v22 )
  {
    if ( v24 >= v22 )
    {
      v33 = v24 - v22;
      if ( v24 > *((unsigned int *)a1 + 81) )
      {
        v6 = (unsigned __int64)(a1 + 41);
        sub_C8D5F0((__int64)(a1 + 39), a1 + 41, (unsigned int)(v20 + 63) >> 6, 8u, v24, v19);
        v22 = *((unsigned int *)a1 + 80);
      }
      if ( 8 * v33 )
      {
        v6 = 0;
        memset((void *)(a1[39] + 8 * v22), 0, 8 * v33);
        LODWORD(v22) = *((_DWORD *)a1 + 80);
      }
      v23 = *((_DWORD *)a1 + 94);
      *((_DWORD *)a1 + 80) = v33 + v22;
    }
    else
    {
      *((_DWORD *)a1 + 80) = (unsigned int)(v20 + 63) >> 6;
    }
  }
  v25 = v23 & 0x3F;
  if ( v25 )
  {
    v6 = *((unsigned int *)a1 + 80);
    *(_QWORD *)(a1[39] + 8 * v6 - 8) &= ~(-1LL << v25);
  }
  v26 = (void *)sub_2207820(8 * v20);
  v27 = v26;
  if ( v26 && v20 )
  {
    v6 = 0;
    v27 = memset(v26, 0, 8 * v20);
  }
  v28 = a1[38];
  a1[38] = (__int64)v27;
  if ( v28 )
    j_j___libc_free_0_0(v28);
  ++a1[1];
  if ( *((_BYTE *)a1 + 36) )
    goto LABEL_37;
  v29 = 4 * (*((_DWORD *)a1 + 7) - *((_DWORD *)a1 + 8));
  v30 = *((unsigned int *)a1 + 6);
  if ( v29 < 0x20 )
    v29 = 32;
  if ( (unsigned int)v30 <= v29 )
  {
    memset((void *)a1[2], -1, 8 * v30);
LABEL_37:
    *(__int64 *)((char *)a1 + 28) = 0;
    return;
  }
  sub_C8C990((__int64)(a1 + 1), v6);
}
