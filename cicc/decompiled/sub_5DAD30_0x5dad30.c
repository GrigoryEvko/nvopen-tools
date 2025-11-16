// Function: sub_5DAD30
// Address: 0x5dad30
//
__int64 __fastcall sub_5DAD30(const char *a1, __int64 a2)
{
  int v2; // r13d
  __int64 v3; // r12
  __int64 result; // rax
  unsigned int v5; // ebx
  int v6; // r14d
  int v7; // r14d
  int v8; // edi
  char *v9; // r15
  FILE *v10; // rsi
  char v11; // al
  int v12; // ebx
  char *v13; // rdx
  int v14; // edi
  char *v15; // r15
  char *v16; // rbx
  int v17; // edi
  __int64 i; // rbx
  __int64 v19; // r15
  char *v20; // r15
  int v21; // edi
  char *v22; // r15
  unsigned __int64 v23; // rdi
  __int64 v24; // rdx
  unsigned __int8 v25; // si
  int v26; // edi
  __int64 v27; // rdi
  int v28; // eax
  __int64 j; // rax
  __int64 v30; // rsi
  int v31; // r13d
  __int64 v32; // rdi
  char *v33; // rbx
  FILE *v34; // rsi
  __int64 v35; // rsi
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // r15
  int v38; // edi
  char *v39; // rbx
  int v40; // eax
  int v41; // edi
  char *v42; // rbx
  unsigned __int64 v43; // rdi
  int v44; // edi
  char *v45; // rbx
  char *v46; // rbx
  int v47; // edi
  __int64 v48; // r15
  __int64 v49; // rsi
  __int64 v50; // rax
  int v51; // edi
  const char *v52; // rbx
  int v53; // edi
  const char *v54; // rbx
  int v55; // [rsp+4h] [rbp-3Ch] BYREF
  __int64 v56[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = a2;
  v3 = (__int64)a1;
  result = qword_4CF7C88;
  v56[0] = 0;
  v55 = 0;
  ++qword_4CF7C88;
  if ( (a1[88] & 8) == 0 )
  {
    a2 = (unsigned int)dword_4CF7EA0;
    if ( !dword_4CF7EA0 )
    {
      qword_4CF7C88 = result;
      if ( result )
        return result;
      goto LABEL_4;
    }
    a1 = "#if 0";
    sub_5D3EB0("#if 0");
    if ( dword_4CF7F40 )
      sub_5D37C0("#if 0", a2);
    dword_4CF7F3C = 0;
    dword_4CF7F44 = 0;
    qword_4CF7F48 = 0;
    dword_4F07508[0] = 0;
    LOWORD(dword_4F07508[1]) = 0;
  }
  v5 = *(_DWORD *)(v3 + 184);
  v6 = 0;
  if ( v5 && unk_4F072D4 != v5 && (!unk_4F068C4 || v5 != 1 || (*(_BYTE *)(v3 + 179) & 0x20) == 0) )
  {
    v7 = dword_4CF7F38;
    if ( dword_4CF7F40 )
      sub_5D37C0(a1, a2);
    ++dword_4CF7F60;
    v8 = 35;
    v9 = "pragma pack(";
    dword_4CF7F38 = 0;
    do
    {
      ++v9;
      putc(v8, stream);
      v8 = *(v9 - 1);
      ++dword_4CF7F40;
    }
    while ( (_BYTE)v8 );
    sub_5D32F0(v5);
    v10 = stream;
    putc(41, stream);
    ++dword_4CF7F40;
    --dword_4CF7F60;
    sub_5D37C0(41, v10);
    dword_4CF7F38 = v7;
    v6 = 1;
  }
  if ( *(char *)(v3 + 88) < 0 )
  {
    v48 = 0;
    while ( 1 )
    {
      v49 = unk_4F04C50;
      if ( unk_4F04C50 )
        v49 = qword_4CF7E98;
      v50 = sub_732D20(v3, v49, 0, v48);
      v48 = v50;
      if ( !v50 )
        break;
      sub_5D52E0(v50, v49);
    }
  }
  sub_5D45D0((unsigned int *)(v3 + 64));
  v11 = *(_BYTE *)(v3 + 140);
  if ( v11 == 10 )
  {
    v12 = 6;
    v13 = "struct";
    goto LABEL_21;
  }
  v12 = 5;
  v13 = "union";
  if ( v11 == 11 )
  {
LABEL_21:
    v14 = *v13;
    v15 = v13 + 1;
    goto LABEL_22;
  }
  if ( v11 != 2 )
    sub_721090(v3 + 64);
  v12 = 4;
  v14 = 101;
  v15 = "num";
  do
  {
LABEL_22:
    ++v15;
    putc(v14, stream);
    v14 = *(v15 - 1);
  }
  while ( *(v15 - 1) );
  dword_4CF7F40 += v12;
  v16 = "{";
  putc(32, stream);
  ++dword_4CF7F40;
  sub_5D71E0(v3);
  v17 = 32;
  do
  {
    ++v16;
    putc(v17, stream);
    v17 = *(v16 - 1);
  }
  while ( *(v16 - 1) );
  dword_4CF7F40 += 2;
  if ( dword_4CF7EA0 )
  {
    putc(32, stream);
    v40 = dword_4CF7EA4;
    ++dword_4CF7F40;
    ++dword_4CF7EA4;
    if ( !v40 )
    {
      v51 = 47;
      v52 = "*";
      do
      {
        ++v52;
        putc(v51, stream);
        v51 = *(v52 - 1);
        ++dword_4CF7F40;
      }
      while ( (_BYTE)v51 );
    }
    v41 = 32;
    v42 = "alignment = ";
    do
    {
      ++v42;
      putc(v41, stream);
      v41 = *(v42 - 1);
    }
    while ( *(v42 - 1) );
    v43 = *(unsigned int *)(v3 + 136);
    dword_4CF7F40 += 13;
    sub_5D32F0(v43);
    putc(32, stream);
    ++dword_4CF7F40;
    if ( !--dword_4CF7EA4 )
    {
      v53 = 42;
      v54 = "/";
      do
      {
        ++v54;
        putc(v53, stream);
        v53 = *(v54 - 1);
        ++dword_4CF7F40;
      }
      while ( (_BYTE)v53 );
    }
    putc(32, stream);
    ++dword_4CF7F40;
  }
  dword_4CF7F38 += 2;
  sub_5DA0F0(v3, v56, &v55);
  if ( v55 )
  {
    for ( i = *(_QWORD *)(v3 + 160); i; ++dword_4CF7F40 )
    {
      while ( 1 )
      {
        if ( *(_QWORD *)(i + 184) )
        {
          putc(32, stream);
          v19 = *(_QWORD *)(i + 184);
          ++dword_4CF7F40;
          sub_74A390(v19, 0, 0, 0, 0, &qword_4CF7CE0);
          sub_74D110(v19, 0, 0, &qword_4CF7CE0);
          putc(32, stream);
          ++dword_4CF7F40;
          sub_5D34A0();
          putc(59, stream);
          ++dword_4CF7F40;
        }
        if ( *(_QWORD *)(i + 176) > (unsigned __int64)*(unsigned __int8 *)(i + 137) )
          break;
        i = *(_QWORD *)(i + 112);
        if ( !i )
          goto LABEL_38;
      }
      v20 = "truct { ";
      putc(32, stream);
      ++dword_4CF7F40;
      v21 = 115;
      do
      {
        ++v20;
        putc(v21, stream);
        v21 = *(v20 - 1);
      }
      while ( *(v20 - 1) );
      dword_4CF7F40 += 9;
      v22 = "} ";
      sub_5D4160(i);
      putc(58, stream);
      v23 = *(unsigned __int8 *)(i + 137);
      ++dword_4CF7F40;
      sub_5D32F0(v23);
      putc(59, stream);
      LOBYTE(v23) = *(_BYTE *)(i + 136);
      v24 = *(_QWORD *)(i + 176);
      v25 = *(_BYTE *)(i + 137);
      ++dword_4CF7F40;
      sub_5D3D20(v23, v25, v24);
      v26 = 32;
      do
      {
        ++v22;
        putc(v26, stream);
        v26 = *(v22 - 1);
      }
      while ( *(v22 - 1) );
      dword_4CF7F40 += 3;
      ++dword_4CF7F60;
      putc(95, stream);
      ++dword_4CF7F40;
      sub_5D34A0();
      --dword_4CF7F60;
      putc(59, stream);
      i = *(_QWORD *)(i + 112);
    }
  }
LABEL_38:
  v27 = v56[0];
  if ( v56[0] )
  {
    if ( *(_BYTE *)(v3 + 140) != 11 )
    {
      v28 = sub_8D3410(*(_QWORD *)(v56[0] + 120));
      v27 = v56[0];
      if ( v28 )
      {
        for ( j = *(_QWORD *)(v56[0] + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        if ( !*(_QWORD *)(j + 128) )
          goto LABEL_44;
      }
    }
    v36 = sub_5D35E0(v27);
    v37 = *(_QWORD *)(v3 + 128);
    if ( *(_BYTE *)(v3 + 140) == 11 )
    {
      if ( v37 <= v36 )
        goto LABEL_44;
    }
    else
    {
      v37 -= v36;
    }
  }
  else
  {
    v37 = *(_QWORD *)(v3 + 128);
  }
  if ( v37 > 1 )
  {
    v44 = 99;
    v45 = "har __nv_no_debug_dummy_end_padding_";
    do
    {
      ++v45;
      putc(v44, stream);
      v44 = *(v45 - 1);
    }
    while ( *(v45 - 1) );
    dword_4CF7F40 += 37;
    v46 = ";";
    sub_5D32F0(qword_4CF7C80);
    ++qword_4CF7C80;
    putc(91, stream);
    ++dword_4CF7F40;
    sub_5D32F0(v37);
    v47 = 93;
    do
    {
      ++v46;
      putc(v47, stream);
      v47 = *(v46 - 1);
    }
    while ( *(v46 - 1) );
    dword_4CF7F40 += 2;
    goto LABEL_47;
  }
  if ( v37 == 1 )
    goto LABEL_63;
LABEL_44:
  if ( (!unk_4F072D8 || !unk_4F068C4) && !sub_72FD90(*(_QWORD *)(v3 + 160), 11) )
  {
LABEL_63:
    if ( *(_QWORD *)(v3 + 128) <= 1u && unk_4F068A0 )
    {
      *(_BYTE *)(v3 + 142) |= 0x10u;
    }
    else
    {
      v38 = 99;
      v39 = "har __nv_no_debug_dummy_end_padding_";
      do
      {
        ++v39;
        putc(v38, stream);
        v38 = *(v39 - 1);
      }
      while ( *(v39 - 1) );
      dword_4CF7F40 += 37;
      sub_5D32F0(qword_4CF7C80);
      ++qword_4CF7C80;
      putc(59, stream);
      ++dword_4CF7F40;
    }
  }
LABEL_47:
  dword_4CF7F38 -= 2;
  putc(125, stream);
  v30 = 1;
  ++dword_4CF7F40;
  sub_74F590(v3, 1, &qword_4CF7CE0);
  if ( v2 )
  {
    v30 = (__int64)stream;
    putc(59, stream);
    ++dword_4CF7F40;
  }
  if ( v6 )
  {
    v31 = dword_4CF7F38;
    if ( dword_4CF7F40 )
      sub_5D37C0((unsigned int)dword_4CF7F40, v30);
    ++dword_4CF7F60;
    LODWORD(v32) = 35;
    v33 = "pragma pack()";
    dword_4CF7F38 = 0;
    do
    {
      v34 = stream;
      ++v33;
      putc(v32, stream);
      v32 = (unsigned int)*(v33 - 1);
      ++dword_4CF7F40;
    }
    while ( (_BYTE)v32 );
    --dword_4CF7F60;
    sub_5D37C0(v32, v34);
    dword_4CF7F38 = v31;
  }
  v35 = (unsigned int)dword_4CF7EA0;
  if ( dword_4CF7EA0 && (*(_BYTE *)(v3 + 88) & 8) == 0 )
  {
    sub_5D3EB0("#endif");
    if ( dword_4CF7F40 )
      sub_5D37C0("#endif", v35);
    dword_4CF7F3C = 0;
    dword_4CF7F44 = 0;
    qword_4CF7F48 = 0;
    dword_4F07508[0] = 0;
    LOWORD(dword_4F07508[1]) = 0;
  }
  *(_BYTE *)(v3 + 142) |= 0x20u;
  result = qword_4CF7C88 - 1;
  qword_4CF7C88 = result;
  if ( !result )
LABEL_4:
    qword_4CF7C80 = 0;
  return result;
}
