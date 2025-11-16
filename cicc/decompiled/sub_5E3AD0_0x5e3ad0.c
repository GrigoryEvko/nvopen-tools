// Function: sub_5E3AD0
// Address: 0x5e3ad0
//
__int64 __fastcall sub_5E3AD0(__int64 a1, void (__fastcall *a2)(__int64))
{
  __int64 result; // rax
  __int64 i; // rax
  bool v6; // zf
  __int64 v7; // rdx
  char *v8; // r15
  __int64 v9; // rcx
  __int64 v10; // rdi
  const char *v11; // rax
  __int64 v12; // rdi
  FILE *v13; // rcx
  const char *v14; // rax
  struct _IO_FILE *v15; // rax
  FILE *v16; // r12
  __int64 v17; // rax
  unsigned int v18; // r12d
  __int64 v19; // rsi
  FILE *v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 j; // rbx
  __int64 v25; // rdx
  __int64 *v26; // rbx
  __int64 v27; // r15
  __int64 v28; // rdx
  __int64 k; // rbx
  __int64 m; // rbx
  __int64 n; // rbx
  FILE *v32; // r12
  const char *v33; // rax
  __int64 v34; // r12
  char *v35; // rax
  __int64 v36; // r12
  char *v37; // rax
  void *v38; // rax
  void *v39; // r12
  void *v40; // rax
  char v41; // bl
  FILE *v42; // [rsp+8h] [rbp-E8h]
  int v43; // [rsp+1Ch] [rbp-D4h]
  _QWORD *v44; // [rsp+20h] [rbp-D0h]
  _QWORD v45[24]; // [rsp+30h] [rbp-C0h] BYREF

  if ( !unk_4D046B8 )
  {
    result = dword_4D045A0;
    if ( !dword_4D045A0 )
      return result;
LABEL_46:
    exit(0);
  }
  unk_4D045D8("Generating NVVM IR", byte_3F871B3);
  if ( unk_4D04530 )
  {
    for ( i = *(_QWORD *)(unk_4F07288 + 144LL); i; i = *(_QWORD *)(i + 112) )
    {
      if ( (*(_QWORD *)(i + 192) & 0x10000000000200LL) == 0x200 )
        *(_DWORD *)(i + 196) |= 0x100400u;
    }
  }
  v43 = unk_4D04630;
  if ( !unk_4D04630 && unk_4D04578 )
  {
    qword_4CF7F58 = 0;
    sub_7461E0(&qword_4CF7CE0);
    byte_4CF7D68 = 1;
    qword_4CF7CE0 = (__int64)sub_5D3290;
    qword_4CF7CE8 = (__int64)sub_5D3250;
    qword_4CF7CF8 = (__int64)sub_5DB900;
    qword_4CF7D18 = (__int64)sub_5D34A0;
    qword_4CF7D20 = (__int64)sub_5D7F00;
    qword_4CF7D28 = (__int64)sub_5DEB60;
    qword_4CF7D40 = (__int64)sub_5D2E00;
    byte_4CF7D6B = 1;
    unk_4CF7D6C = 256;
    byte_4CF7CD2 = 3;
    byte_4CF7D69 = unk_4F072D0;
    byte_4CF7CD1 = 3;
    byte_4CF7CD0 = 3;
    qword_4CF7CC8 = 0;
    qword_4CF7CC0 = 0;
    v6 = *qword_4F076F0 == 45;
    qword_4CF7CB8 = 0;
    qword_4CF7CB0 = 0;
    qword_4CF7CA8 = 0;
    qword_4CF7CA0 = 0;
    if ( !v6 || (v7 = 0, v8 = 0, qword_4F076F0[1]) )
    {
      v7 = 0;
      v8 = (char *)unk_4D04510;
      if ( !unk_4D04510 )
      {
        v17 = sub_722560(qword_4F076F0, ".int.c");
        v7 = qword_4CF7CC8;
        v8 = (char *)v17;
      }
    }
    dword_4CF7F60 = 0;
    stream = 0;
    dword_4F07508[0] = 0;
    LOWORD(dword_4F07508[1]) = 0;
    qword_4CF7F48 = 0;
    unk_4F04C50 = 0;
    dword_4CF7F44 = 0;
    dword_4CF7F40 = 0;
    dword_4CF7F3C = 0;
    dword_4CF7F38 = 0;
    dword_4CF7EA4 = 0;
    dword_4CF7EA0 = 0;
    qword_4CF7EA8 = 0;
    qword_4CF7EB8 = 0;
    qword_4CF7EB0 = 0;
    dword_4CF7CD4 = 0;
    qword_4CF7E98 = 0;
    qword_4CF7E90 = (__int64)&unk_4CF7DC0;
    qword_4CF7DA0 = 0;
    qword_4CF7D98 = 0;
    qword_4CF7D90 = 0;
    dword_4CF7D8C = 0;
    dword_4CF7D88 = 0;
    if ( v7 )
    {
      v9 = qword_4CF7CB8;
      qword_4CF7CC8 = 0;
      qword_4CF7CB8 = v7;
      *(_QWORD *)qword_4CF7CC0 = v9;
      qword_4CF7CC0 = 0;
    }
    if ( !v8 )
    {
      v15 = stdout;
      stream = stdout;
      goto LABEL_50;
    }
    stream = (FILE *)sub_685E40(v8, 0, 0, 0, 1700);
    v10 = unk_4D04580;
    if ( !unk_4D04580 )
    {
      v36 = sub_8237A0(256);
      unk_4D04580 = sub_722560(v8, ".device.c");
      v37 = (char *)sub_722430(v8);
      sub_720CF0(v37);
      v10 = *(_QWORD *)(v36 + 32);
      unk_4D04580 = v10;
    }
    qword_4CF7EB8 = (FILE *)sub_685E40(v10, 0, 0, 0, 1700);
    v11 = (const char *)sub_7462A0(unk_4F06B39, 1);
    fprintf(qword_4CF7EB8, "typedef %s __nv_bool;\n", v11);
    v12 = unk_4D04578;
    qword_4CF7EE0 = 0;
    qword_4CF7EE8 = 0;
    dword_4CF7EF0 = 0;
    if ( !unk_4D04578 )
    {
      v34 = sub_8237A0(256);
      unk_4D04578 = sub_722560(v8, ".stub.c");
      v35 = (char *)sub_722430(v8);
      sub_720CF0(v35);
      v12 = *(_QWORD *)(v34 + 32);
      unk_4D04578 = v12;
    }
    qword_4CF7EB0 = (FILE *)sub_685E40(v12, 0, 0, 0, 1700);
    v13 = qword_4CF7EB0;
    if ( unk_4F068D0 )
      goto LABEL_81;
    if ( unk_4D0455C )
      goto LABEL_25;
    if ( unk_4F068D8 > 0x9E97u )
    {
LABEL_81:
      fwrite("#pragma GCC diagnostic push\n", 1u, 0x1Cu, qword_4CF7EB0);
      v13 = qword_4CF7EB0;
      if ( unk_4F068D0 )
      {
        fwrite("#pragma GCC diagnostic ignored \"-Wmismatched-tags\"\n", 1u, 0x33u, qword_4CF7EB0);
        v13 = qword_4CF7EB0;
LABEL_24:
        fwrite("#pragma GCC diagnostic ignored \"-Wunused-function\"\n", 1u, 0x33u, v13);
        fwrite("#pragma GCC diagnostic ignored \"-Wcast-qual\"\n", 1u, 0x2Du, qword_4CF7EB0);
        v13 = qword_4CF7EB0;
        goto LABEL_25;
      }
      if ( unk_4D0455C )
      {
LABEL_25:
        if ( qword_4D045BC
          && (fwrite("#define __NV_MODULE_ID ", 1u, 0x17u, v13),
              v32 = qword_4CF7EB0,
              v33 = (const char *)sub_723F40(0),
              fputs(v33, v32),
              fputc(10, qword_4CF7EB0),
              v13 = qword_4CF7EB0,
              HIDWORD(qword_4D045BC)) )
        {
          fwrite("#define __NV_CUBIN_HANDLE_STORAGE__ extern\n", 1u, 0x2Bu, qword_4CF7EB0);
        }
        else
        {
          fwrite("#define __NV_CUBIN_HANDLE_STORAGE__ static\n", 1u, 0x2Bu, v13);
        }
        fwrite("#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)\n", 1u, 0x39u, qword_4CF7EB0);
        fwrite("#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__\n", 1u, 0x33u, qword_4CF7EB0);
        fwrite("#endif\n", 1u, 7u, qword_4CF7EB0);
        fwrite("#include \"crt/host_runtime.h\"\n", 1u, 0x1Eu, qword_4CF7EB0);
        v14 = (const char *)sub_7212A0(unk_4D04588);
        fprintf(qword_4CF7EB0, "#include \"%s\"\n", v14);
        v15 = stream;
LABEL_50:
        qword_4CF7F58 = (__int64)v15;
        v18 = 1;
        v19 = sub_729EA0();
        sub_5D43B0(*(_DWORD *)(v19 + 32), v19);
        v20 = qword_4CF7EB8;
        v44 = (_QWORD *)unk_4F07288;
        qword_4CF7E98 = unk_4F07288;
        v42 = stream;
        sub_5D3B20(qword_4CF7EB8);
        while ( 1 )
        {
          for ( j = v44[13]; j; j = *(_QWORD *)(j + 112) )
          {
            while ( 1 )
            {
              v20 = (FILE *)j;
              if ( !(unsigned int)sub_736DD0(j) )
                break;
              j = *(_QWORD *)(j + 112);
              if ( !j )
                goto LABEL_56;
            }
            v20 = (FILE *)j;
            v19 = v18;
            sub_5DB980((FILE *)j, v18, v25, v21, v22, v23);
          }
LABEL_56:
          if ( unk_4F072C8 == 1 && !v43 )
          {
            LODWORD(v45[0]) = 0;
            for ( k = v44[18]; k; k = *(_QWORD *)(k + 112) )
            {
              v20 = *(FILE **)(k + 152);
              v19 = v18;
              sub_5E3A50((__int64)v20, v18, v45, v21, v22, v23);
            }
            for ( m = v44[14]; m; m = *(_QWORD *)(m + 112) )
            {
              v20 = *(FILE **)(m + 120);
              v19 = v18;
              sub_5E3A50((__int64)v20, v18, v45, v21, v22, v23);
            }
            for ( n = v44[15]; n; n = *(_QWORD *)(n + 112) )
            {
              v20 = *(FILE **)(n + 120);
              v19 = v18;
              sub_5E3A50((__int64)v20, v18, v45, v21, v22, v23);
            }
            v43 = LODWORD(v45[0]) == 0;
          }
          v26 = (__int64 *)unk_4F072C0;
          if ( unk_4F072C0 )
            break;
LABEL_71:
          if ( v18 == 2 )
          {
            if ( dword_4CF7F40 )
              sub_5D37C0(v20, v19);
            if ( v42 != stream )
              sub_5D3B20(v42);
            sub_5E3680((__int64)v44, 0);
            sub_5DFB80(v44, 1, 1u, 0);
            v16 = stream;
            sub_5D3B20(qword_4CF7EB8);
            sub_5D3EB0("#include \"common_functions.h\"");
            if ( dword_4CF7F40 )
              sub_5D37C0("#include \"common_functions.h\"", 1);
            if ( v16 != stream )
              sub_5D3B20(v16);
            sub_5DFB80(v44, 0, 0, 1u);
            sub_5E3680((__int64)v44, 1u);
            if ( dword_4CF7F40 )
              sub_5D37C0(v44, 1);
            sub_686700(&stream, 1700);
            qword_4CF7F58 = 0;
            if ( unk_4F068D0 || !unk_4D0455C && unk_4F068D8 > 0x9E97u )
              fwrite("\n#pragma GCC diagnostic pop\n", 1u, 0x1Cu, qword_4CF7EB0);
            if ( unk_4D045A8 )
              sub_724450();
            goto LABEL_44;
          }
          v18 = 2;
        }
        while ( 1 )
        {
          if ( (*(_DWORD *)(v26[1] + 192) & 0x8000400) == 0 )
          {
            v27 = v26[3];
            if ( v27 )
              break;
          }
LABEL_59:
          v26 = (__int64 *)*v26;
          if ( !v26 )
            goto LABEL_71;
        }
        while ( 1 )
        {
          v20 = (FILE *)v27;
          if ( (unsigned int)sub_736DD0(v27) )
            goto LABEL_65;
          if ( *(_BYTE *)(v27 + 140) == 12 )
            break;
          if ( v18 == 1 )
            goto LABEL_70;
LABEL_64:
          v19 = v18;
          v20 = (FILE *)v27;
          sub_5DB980((FILE *)v27, v18, v28, v21, v22, v23);
LABEL_65:
          v27 = *(_QWORD *)(v27 + 112);
          if ( !v27 )
            goto LABEL_59;
        }
        if ( (*(_BYTE *)(v27 + 186) & 1) != 0 )
          goto LABEL_65;
        if ( v18 != 1 )
          goto LABEL_64;
LABEL_70:
        sub_5E3920(v27, v26[1], *((_DWORD *)v26 + 4));
        goto LABEL_64;
      }
    }
    if ( unk_4F068D8 <= 0x9D07u )
      goto LABEL_25;
    goto LABEL_24;
  }
LABEL_44:
  if ( dword_4D045A0 )
  {
    v38 = dlopen("libTileIRCompiler_shared.so", 1);
    v39 = v38;
    if ( !v38 )
    {
      printf("\n error: unable to open %s!\n", "libTileIRCompiler_shared.so");
      exit(1);
    }
    v40 = dlsym(v38, "cudacc_back_end");
    if ( !v40 )
    {
      printf("\n error: unable to lookup tile handler function!");
      exit(1);
    }
    memset(&v45[2], 0, 24);
    v45[9] = sub_5D2E60;
    v45[0] = sub_620FD0;
    v45[10] = sub_5D3760;
    v45[1] = sub_620FA0;
    v45[11] = sub_5D3700;
    v45[7] = sub_72F9F0;
    v45[13] = sub_5D2EA0;
    v45[5] = sub_72B0F0;
    v45[14] = sub_5D2EB0;
    v45[12] = sub_622850;
    v45[15] = sub_729B10;
    v45[16] = sub_686610;
    v45[6] = 0;
    v45[8] = 0;
    v41 = ((__int64 (__fastcall *)(_QWORD, void *, _QWORD *))v40)(unk_4D04590, &unk_4F07280, v45);
    dlclose(v39);
    if ( !v41 )
    {
      printf("\n failed tileir gen!");
      exit(1);
    }
    goto LABEL_46;
  }
  a2(a1);
  unk_4D045D0();
  result = dword_4D045A0;
  if ( dword_4D045A0 )
    goto LABEL_46;
  return result;
}
