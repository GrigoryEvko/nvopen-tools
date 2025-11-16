// Function: sub_5E0F20
// Address: 0x5e0f20
//
__int64 __fastcall sub_5E0F20(__int64 a1, FILE *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  _QWORD *v8; // r13
  __int64 v9; // r15
  int v10; // r13d
  int v11; // r12d
  int v12; // eax
  __int64 v13; // rcx
  int v14; // r12d
  __int64 j; // rax
  __int64 v17; // rdx
  __int64 v19; // r15
  _QWORD *i; // r15
  __int64 v21; // r15
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  FILE *v26; // r13
  FILE *v27; // rdi
  int v28; // eax
  int v29; // edi
  char *v30; // r14
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  int *v35; // rax
  __int64 v36; // [rsp+0h] [rbp-110h]
  __int64 v37; // [rsp+8h] [rbp-108h]
  _QWORD v38[10]; // [rsp+10h] [rbp-100h] BYREF
  int v39; // [rsp+64h] [rbp-ACh]

  v37 = qword_4CF7E98;
  v7 = qword_4CF7E90;
  qword_4CF7E90 = (__int64)&unk_4CF7DC0;
  v36 = v7;
  v8 = (_QWORD *)unk_4F04C50;
  if ( a1 != *(_QWORD *)(unk_4F04C50 + 80LL) )
  {
    v8 = *(_QWORD **)(*(_QWORD *)(a1 + 80) + 8LL);
    if ( !v8 )
    {
LABEL_3:
      v9 = *(_QWORD *)(a1 + 72);
      if ( !v9 )
        goto LABEL_13;
      goto LABEL_4;
    }
  }
  v19 = v8[17];
  qword_4CF7E98 = (__int64)v8;
  if ( v19 )
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(v19 + 120) & 0x20) != 0 )
      {
        if ( (*(_BYTE *)(v19 + 88) & 8) != 0 )
          goto LABEL_41;
        if ( dword_4CF7EA0 )
          break;
      }
LABEL_24:
      v19 = *(_QWORD *)(v19 + 112);
      if ( !v19 )
        goto LABEL_17;
    }
    sub_5D3EB0("#if 0");
    if ( dword_4CF7F40 )
      sub_5D37C0("#if 0", a2);
    dword_4CF7F3C = 0;
    dword_4CF7F44 = 0;
    qword_4CF7F48 = 0;
    dword_4F07508[0] = 0;
    LOWORD(dword_4F07508[1]) = 0;
LABEL_41:
    v29 = 95;
    v30 = "_label__ ";
    do
    {
      ++v30;
      putc(v29, stream);
      v29 = *(v30 - 1);
    }
    while ( *(v30 - 1) );
    dword_4CF7F40 += 10;
    sub_5D5A80(v19, 0);
    a2 = stream;
    putc(59, stream);
    ++dword_4CF7F40;
    sub_5D37C0(59, a2);
    if ( dword_4CF7EA0 && (*(_BYTE *)(v19 + 88) & 8) == 0 )
    {
      sub_5D3EB0("#endif");
      if ( dword_4CF7F40 )
        sub_5D37C0("#endif", a2);
      dword_4CF7F3C = 0;
      dword_4CF7F44 = 0;
      qword_4CF7F48 = 0;
      dword_4F07508[0] = 0;
      LOWORD(dword_4F07508[1]) = 0;
    }
    goto LABEL_24;
  }
LABEL_17:
  for ( i = (_QWORD *)v8[29]; i; i = (_QWORD *)*i )
  {
    if ( !i[3] )
      sub_5D52E0((__int64)i, (__int64)a2);
  }
  sub_5DFB80(v8, 0, 1u, 1u);
  v21 = *(_QWORD *)(a1 + 72);
  sub_76C7C0(v38, 0, v22, v23, v24, v25);
  a2 = (FILE *)v38;
  v38[0] = sub_5D5490;
  v38[6] = sub_5D3A80;
  v38[2] = nullsub_1;
  v39 = 1;
  sub_76D840(v21, v38);
  v26 = qword_4CF7EA8;
  if ( qword_4CF7EA8 )
  {
    v27 = qword_4CF7EA8;
    if ( fseek(qword_4CF7EA8, 0, 0) )
    {
      v35 = __errno_location();
      sub_6866A0(1512, (unsigned int)*v35);
    }
    if ( dword_4CF7F40 )
      sub_5D37C0(v27, (unsigned int)dword_4CF7F40);
    while ( 1 )
    {
      v28 = getc(v26);
      a2 = stream;
      if ( v28 == -1 )
        break;
      putc(v28, stream);
    }
    fputc(10, stream);
    if ( dword_4CF7F40 )
      sub_5D37C0(10, a2);
    dword_4CF7F3C = 0;
    dword_4CF7F44 = 0;
    qword_4CF7F48 = 0;
    dword_4F07508[0] = 0;
    LOWORD(dword_4F07508[1]) = 0;
    j__fclose(v26);
    qword_4CF7EA8 = 0;
  }
  v9 = *(_QWORD *)(a1 + 72);
  if ( !v9 )
    goto LABEL_13;
  if ( !*(_DWORD *)v9 )
  {
    sub_5D45D0((unsigned int *)a1);
    goto LABEL_3;
  }
LABEL_4:
  v10 = 0;
  v11 = 0;
  do
  {
    while ( 1 )
    {
      v12 = *(unsigned __int8 *)(v9 + 40);
      v13 = (unsigned int)(v12 - 20);
      if ( (unsigned __int8)(v12 - 20) <= 2u )
        break;
      v11 = 1;
      sub_5DFD00(v9, a2, a3, v13, a5, a6);
      v9 = *(_QWORD *)(v9 + 16);
      if ( !v9 )
        goto LABEL_11;
    }
    if ( (_BYTE)v12 == 22 && v11 )
    {
      a2 = stream;
      ++v10;
      v11 = 0;
      putc(123, stream);
      ++dword_4CF7F40;
      sub_5DFD00(v9, a2, v31, v32, v33, v34);
    }
    else
    {
      sub_5DFD00(v9, a2, a3, v13, a5, a6);
    }
    v9 = *(_QWORD *)(v9 + 16);
  }
  while ( v9 );
LABEL_11:
  v14 = v10 - 1;
  if ( v10 )
  {
    do
    {
      putc(125, stream);
      ++dword_4CF7F40;
    }
    while ( v14-- != 0 );
  }
LABEL_13:
  qword_4CF7E98 = v37;
  for ( j = qword_4CF7E90; (_UNKNOWN *)j != &unk_4CF7DC0; *(_QWORD *)(v17 + 160) = 0 )
  {
    v17 = j;
    j = *(_QWORD *)(j + 160);
  }
  qword_4CF7E90 = v36;
  return v36;
}
