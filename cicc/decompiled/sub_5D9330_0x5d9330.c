// Function: sub_5D9330
// Address: 0x5d9330
//
__int64 __fastcall sub_5D9330(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r15
  __int64 v5; // r13
  unsigned __int8 v6; // bl
  FILE *v7; // r12
  FILE *v8; // rsi
  _BOOL8 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r9
  _BYTE *v12; // r8
  __int64 result; // rax
  unsigned __int8 v14; // r14
  __int64 i; // rdx
  int v16; // r14d
  char v17; // al
  char v18; // al
  char *v19; // rbx
  char v20; // al
  char *v21; // rbx
  bool v22; // cf
  bool v23; // zf
  bool v24; // cf
  bool v25; // zf
  int v26; // eax
  char v27; // al
  char *v28; // rbx
  const char *v29; // rdi
  int v30; // ebx
  char v31; // al
  const char *v32; // r14
  __int64 v33; // r8
  __int64 j; // r13
  char v35; // bl
  char v36; // al
  char *v37; // rbx
  int v38; // eax
  char v39; // al
  char *v40; // rbx
  __int64 v41; // r12
  __int64 v42; // rsi
  __int64 v43; // rax
  bool v44; // al
  unsigned int v45; // eax
  char v46; // al
  char *v47; // rbx
  char v48; // al
  const char *v49; // rbx
  char v50; // al
  const char *v51; // rbx
  char v52; // al
  char *v53; // rbx
  char v54; // al
  char *v55; // rbx
  char v56; // al
  char *v57; // rbx
  char *v58; // rbx
  FILE *v59; // [rsp+0h] [rbp-140h]
  bool v60; // [rsp+8h] [rbp-138h]
  __int64 v61; // [rsp+8h] [rbp-138h]
  char v62; // [rsp+18h] [rbp-128h]
  int v63; // [rsp+18h] [rbp-128h]
  __int64 v64; // [rsp+20h] [rbp-120h]
  char v65; // [rsp+29h] [rbp-117h]
  char v66; // [rsp+2Ah] [rbp-116h]
  bool v67; // [rsp+2Bh] [rbp-115h]
  int v68; // [rsp+2Ch] [rbp-114h]
  unsigned int v69; // [rsp+2Ch] [rbp-114h]
  unsigned int v70; // [rsp+2Ch] [rbp-114h]
  char v71; // [rsp+3Bh] [rbp-105h] BYREF
  int v72; // [rsp+3Ch] [rbp-104h] BYREF
  FILE v73; // [rsp+40h] [rbp-100h] BYREF

  v3 = a1;
  v5 = *(_QWORD *)(a1 + 120);
  v6 = *(_BYTE *)(a1 + 136);
  v68 = a2;
  v7 = stream;
  if ( !(unsigned int)sub_8D2FF0(v5, a2) || (*(_BYTE *)(a1 + 89) & 1) != 0 )
  {
    v66 = 0;
    v62 = *(_BYTE *)(a1 + 156) & 1;
    if ( !v62 )
      goto LABEL_4;
    goto LABEL_60;
  }
  v66 = *(_BYTE *)(a1 + 156) & 1 | (dword_4CF7C68 != 0);
  if ( v66 )
  {
LABEL_60:
    v7 = stream;
    sub_5D3B20(qword_4CF7EB8);
    v62 = 1;
    goto LABEL_4;
  }
  dword_4CF7C68 = 1;
  sub_5D9330(a1, (unsigned int)a2, a3);
  v62 = 0;
  dword_4CF7C68 = 0;
  v66 = 1;
LABEL_4:
  v8 = (FILE *)qword_4CF7E98;
  sub_72F9F0(a1, qword_4CF7E98, &v71, &v73);
  v64 = 0;
  v11 = 1;
  if ( v71 == 1 )
  {
    v64 = **(_QWORD **)&v73._flags;
    LOBYTE(v11) = **(_QWORD **)&v73._flags == 0;
  }
  v12 = *(_BYTE **)(a1 + 8);
  if ( *(char *)(a1 + 172) < 0 )
  {
    v6 = 1;
    LOBYTE(a3) = 0;
  }
  LOBYTE(v9) = v12 != 0 && (*(_BYTE *)(a1 + 88) & 0x70) == 16;
  v67 = v9;
  if ( v9 )
  {
    v22 = *v12 < 0x5Fu;
    v23 = *v12 == 95;
    if ( *v12 == 95 )
    {
      v10 = 7;
      a1 = (__int64)"__link";
      v8 = *(FILE **)(v3 + 8);
      do
      {
        if ( !v10 )
          break;
        v22 = LOBYTE(v8->_flags) < *(_BYTE *)a1;
        v23 = LOBYTE(v8->_flags) == *(_BYTE *)a1;
        v8 = (FILE *)((char *)v8 + 1);
        ++a1;
        --v10;
      }
      while ( v23 );
      if ( (!v22 && !v23) != v22 )
      {
        v67 = 0;
        v24 = 0;
        v25 = unk_4F068E4 == 0;
        if ( unk_4F068E4 )
        {
          v10 = 19;
          a1 = (__int64)"__builtin_va_alist";
          v8 = *(FILE **)(v3 + 8);
          do
          {
            if ( !v10 )
              break;
            v24 = LOBYTE(v8->_flags) < *(_BYTE *)a1;
            v25 = LOBYTE(v8->_flags) == *(_BYTE *)a1;
            v8 = (FILE *)((char *)v8 + 1);
            ++a1;
            --v10;
          }
          while ( v25 );
          v67 = (!v24 && !v25) == v24;
        }
      }
    }
    else
    {
      v67 = 0;
    }
  }
  if ( (v68 & 1) == 0 && (_BYTE)v11 || (*(_BYTE *)(v3 + 173) & 1) != 0 || (*(_BYTE *)(v3 - 8) & 0x10) != 0 )
    goto LABEL_11;
  if ( v67 || (*(_BYTE *)(v3 + 88) & 8) != 0 )
  {
    v60 = v64 != 0;
    v65 = a3 & (v64 != 0);
    if ( v65 )
    {
      sub_76C7C0(&v73, v8, v9, v10, v12, v11);
      *(_QWORD *)&v73._flags = sub_5D5490;
      v73._IO_write_end = (char *)sub_5D3A80;
      v73._IO_read_end = (char *)nullsub_1;
      HIDWORD(v73._IO_backup_base) = 1;
      sub_76D560(v64, &v73);
      v65 = 1;
      v60 = 1;
    }
  }
  else
  {
    if ( !dword_4CF7EA0 )
    {
LABEL_11:
      result = (unsigned int)dword_4CF7F40;
      goto LABEL_12;
    }
    sub_5D3EB0("#if 0");
    if ( dword_4CF7F40 )
      sub_5D37C0("#if 0", v8);
    dword_4CF7F3C = 0;
    dword_4F07508[0] = 0;
    LOWORD(dword_4F07508[1]) = 0;
    v44 = v64 != 0;
    dword_4CF7F44 = 0;
    qword_4CF7F48 = 0;
    if ( ((unsigned __int8)a3 & (v64 != 0)) != 0 )
      v44 = a3 & (v64 != 0);
    v65 = a3 & (v64 != 0);
    v60 = v44;
  }
  if ( v68 && *(char *)(v3 + 88) < 0 )
  {
    v59 = v7;
    v41 = 0;
    while ( 1 )
    {
      v42 = unk_4F04C50;
      if ( unk_4F04C50 )
        v42 = qword_4CF7E98;
      v43 = sub_732D20(v3, v42, 0, v41);
      v41 = v43;
      if ( !v43 )
        break;
      sub_5D52E0(v43, v42);
    }
    v7 = v59;
  }
  sub_5D45D0((unsigned int *)(v3 + 64));
  if ( !v6 )
  {
    v14 = v68 & (a3 ^ 1);
    if ( v60 )
      v6 = v14;
  }
  if ( unk_4F068C4 && (*(_BYTE *)(v3 + 169) & 8) == 0 )
    v6 = 5;
  for ( i = *(_QWORD *)(v3 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (*(_BYTE *)(v3 + 156) & 2) == 0
    || *(_QWORD *)(i + 128)
    || (v61 = i, !(unsigned int)sub_8D3410(i))
    || *(_QWORD *)(v61 + 176)
    || (v16 = 1, (*(_BYTE *)(v61 + 169) & 0x20) != 0) )
  {
    if ( v62 && (v16 = HIDWORD(qword_4D045BC)) == 0 )
      v6 = 2;
    else
      v16 = 0;
  }
  if ( *(_BYTE *)(v3 + 136) == v6 )
    sub_5D57F0(v3);
  else
    sub_5D3F60(v6);
  if ( !v62 )
  {
    if ( *(_QWORD *)(v3 + 240) )
    {
      if ( unk_4F068C4 )
      {
        v52 = 32;
        v53 = "__attribute__((__weak__))";
        do
        {
          ++v53;
          putc(v52, stream);
          v52 = *(v53 - 1);
        }
        while ( v52 );
        dword_4CF7F40 += 26;
      }
      putc(32, stream);
      v26 = dword_4CF7EA4;
      ++dword_4CF7F40;
      ++dword_4CF7EA4;
      if ( !v26 )
      {
        v50 = 47;
        v51 = "*";
        do
        {
          ++v51;
          putc(v50, stream);
          v50 = *(v51 - 1);
          ++dword_4CF7F40;
        }
        while ( v50 );
      }
      v27 = 32;
      v28 = "COMDAT group: ";
      do
      {
        ++v28;
        putc(v27, stream);
        v27 = *(v28 - 1);
      }
      while ( v27 );
      v29 = *(const char **)(v3 + 240);
      v30 = dword_4CF7F40 + 15;
      dword_4CF7F40 += 15;
      v63 = strlen(v29);
      v31 = *v29;
      v32 = v29 + 1;
      if ( *v29 )
      {
        do
        {
          ++v32;
          putc(v31, stream);
          v31 = *(v32 - 1);
        }
        while ( v31 );
        v30 = dword_4CF7F40;
      }
      dword_4CF7F40 = v30 + v63;
      putc(32, stream);
      ++dword_4CF7F40;
      if ( !--dword_4CF7EA4 )
      {
        v48 = 42;
        v49 = "/";
        do
        {
          ++v49;
          putc(v48, stream);
          v48 = *(v49 - 1);
          ++dword_4CF7F40;
        }
        while ( v48 );
      }
      putc(32, stream);
      ++dword_4CF7F40;
      if ( unk_4F068C4 )
      {
        if ( (v68 & 1) != 0 )
        {
          LOBYTE(v16) = v71 == 3 || v71 == 0;
          goto LABEL_86;
        }
      }
    }
LABEL_85:
    LOBYTE(v16) = 0;
    goto LABEL_86;
  }
  v17 = *(_BYTE *)(v3 + 156);
  if ( HIDWORD(qword_4D045BC)
    && !v6
    && *(_BYTE *)(v3 + 136) != 2
    && (v17 < 0 || ((v17 & 0x40) != 0 || (*(_QWORD *)(v3 + 168) & 0x2000100000LL) != 0) && *(_QWORD *)(v3 + 240)) )
  {
    sub_5D31F0(" __attribute__((nv_weak_odr)) ");
    v17 = *(_BYTE *)(v3 + 156);
  }
  if ( (v17 & 2) != 0 )
  {
    v39 = 32;
    v40 = "__shared__ ";
    do
    {
      ++v40;
      putc(v39, stream);
      v39 = *(v40 - 1);
    }
    while ( v39 );
    goto LABEL_53;
  }
  if ( (v17 & 4) != 0 )
  {
    v56 = 32;
    v57 = "__constant__ ";
    do
    {
      ++v57;
      putc(v56, stream);
      v56 = *(v57 - 1);
    }
    while ( v56 );
    dword_4CF7F40 += 14;
  }
  else
  {
    if ( (v17 & 1) != 0 )
    {
      v18 = 32;
      v19 = "__device__ ";
      do
      {
        ++v19;
        putc(v18, stream);
        v18 = *(v19 - 1);
      }
      while ( v18 );
LABEL_53:
      dword_4CF7F40 += 12;
      goto LABEL_54;
    }
    v23 = (unsigned int)sub_8D3030(v5) == 0;
    v54 = 32;
    if ( v23 )
    {
      v58 = "__text__ ";
      do
      {
        ++v58;
        putc(v54, stream);
        v54 = *(v58 - 1);
      }
      while ( v54 );
    }
    else
    {
      v55 = "__surf__ ";
      do
      {
        ++v55;
        putc(v54, stream);
        v54 = *(v55 - 1);
      }
      while ( v54 );
    }
    dword_4CF7F40 += 10;
  }
LABEL_54:
  if ( (*(_BYTE *)(v3 + 157) & 1) != 0 )
  {
    v46 = 32;
    v47 = "__managed__ ";
    do
    {
      ++v47;
      putc(v46, stream);
      v46 = *(v47 - 1);
    }
    while ( v46 );
    dword_4CF7F40 += 13;
  }
  if ( v16 )
    goto LABEL_85;
  v20 = 32;
  v21 = "__var_used__ ";
  do
  {
    ++v21;
    putc(v20, stream);
    v20 = *(v21 - 1);
  }
  while ( v20 );
  dword_4CF7F40 += 14;
LABEL_86:
  if ( (*(_BYTE *)(v3 + 176) & 8) == 0 && !unk_4F068B8 && (*(_BYTE *)(v3 + 140) & 1) != 0 )
  {
    v36 = 95;
    v37 = "_thread ";
    do
    {
      ++v37;
      putc(v36, stream);
      v36 = *(v37 - 1);
    }
    while ( v36 );
    dword_4CF7F40 += 9;
  }
  if ( (unsigned int)sub_5D3C50(v3) )
  {
    v33 = 1;
  }
  else
  {
    v45 = sub_8D2600(v5);
    v33 = v45;
    if ( v45 )
    {
      v33 = 0;
      if ( (*(_BYTE *)(v5 + 140) & 0xFB) == 8 )
        v33 = sub_8D4C10(v5, unk_4F077C4 != 2) & 1;
    }
  }
  if ( unk_4F077C4 != 2 && unk_4F07778 > 201709 )
  {
    v70 = v33;
    v38 = sub_74A1C0(*(_QWORD *)(v3 + 104), 1, &qword_4CF7CE0);
    v33 = v70;
    if ( v38 )
    {
      putc(32, stream);
      ++dword_4CF7F40;
      v33 = v70;
    }
  }
  v69 = v33;
  sub_74A390(v5, 0, 1, 0, v33, &qword_4CF7CE0);
  sub_5D6390(v3);
  sub_74D110(v5, 0, v69, &qword_4CF7CE0);
  if ( (*(_BYTE *)(v3 + 169) & 8) != 0 )
    sub_7503A0(*(_QWORD *)(v3 + 144), &qword_4CF7CE0);
  else
    sub_750460(*(unsigned __int8 *)(v3 + 144), &qword_4CF7CE0);
  sub_74F860(v3, 1, &qword_4CF7CE0);
  if ( (*(_BYTE *)(v3 + 156) & 1) != 0 && v71 == 3 || v66 )
    goto LABEL_115;
  if ( (v16 & 1) != 0
    || v65
    || v71 == 3
    && *(char *)(v3 + 172) >= 0
    && (*(_BYTE *)(v3 + 136) > 2u
     || !(unsigned int)sub_8D3410(*(_QWORD *)(v3 + 120))
     || (*(_QWORD *)(v3 + 168) & 0x2000100000LL) != 0) )
  {
    for ( j = *(_QWORD *)(v3 + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v35 = byte_4CF7D76;
    v72 = 0;
    if ( v64 && (unsigned int)sub_8D2B80(*(_QWORD *)(v64 + 128)) )
      byte_4CF7D76 = 1;
    memset(&v73, 0, 20);
    sub_5D80F0(v3, j, v64, &v72, 0, &v73);
    byte_4CF7D76 = v35;
    if ( LODWORD(v73._IO_read_ptr) )
      sub_5D4940(v3, &v73);
LABEL_115:
    v8 = stream;
LABEL_116:
    a1 = 59;
    goto LABEL_117;
  }
  v8 = stream;
  if ( !unk_4D045B8 || qword_4CF7EB8 != stream )
    goto LABEL_116;
  a1 = 59;
  if ( *(_BYTE *)(v3 + 136) != 3 )
  {
LABEL_117:
    putc(59, v8);
    ++dword_4CF7F40;
    goto LABEL_118;
  }
  putc(59, stream);
  a1 = v3;
  ++dword_4CF7F40;
  sub_5D68B0(v3, (__int64)v8);
LABEL_118:
  if ( v67 || !dword_4CF7EA0 || (*(_BYTE *)(v3 + 88) & 8) != 0 )
    goto LABEL_11;
  sub_5D3EB0("#endif");
  a1 = (unsigned int)dword_4CF7F40;
  if ( !dword_4CF7F40 )
  {
    result = (__int64)dword_4F07508;
    dword_4CF7F3C = 0;
    dword_4CF7F44 = 0;
    qword_4CF7F48 = 0;
    dword_4F07508[0] = 0;
    LOWORD(dword_4F07508[1]) = 0;
    goto LABEL_14;
  }
  sub_5D37C0((unsigned int)dword_4CF7F40, v8);
  dword_4CF7F3C = 0;
  dword_4CF7F44 = 0;
  result = (unsigned int)dword_4CF7F40;
  qword_4CF7F48 = 0;
  dword_4F07508[0] = 0;
  LOWORD(dword_4F07508[1]) = 0;
LABEL_12:
  if ( (_DWORD)result )
    result = sub_5D37C0(a1, v8);
LABEL_14:
  if ( v7 != stream )
    return sub_5D3B20(v7);
  return result;
}
