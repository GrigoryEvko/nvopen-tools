// Function: sub_5DA0F0
// Address: 0x5da0f0
//
char __fastcall sub_5DA0F0(__int64 a1, __int64 *a2, _DWORD *a3)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r14
  unsigned __int64 v8; // rax
  __int64 v9; // r14
  _BYTE *v10; // rdi
  char v11; // al
  char *v12; // r14
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // r15
  __int64 v19; // rax
  unsigned __int64 v20; // r14
  unsigned __int64 v21; // r14
  __int64 v22; // r15
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 i; // r14
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rsi
  __int64 v28; // rdi
  char *v29; // r14
  char v30; // al
  unsigned __int64 v31; // r14
  int v32; // eax
  char v33; // al
  char *v34; // r15
  char *v35; // rax
  int v36; // r15d
  char *v37; // r14
  char v38; // al
  unsigned __int64 v39; // r14
  char v40; // al
  char *v41; // r14
  FILE *v42; // rsi
  __int64 v43; // rdx
  __int64 v44; // rdi
  unsigned __int64 v45; // rdi
  int v46; // ecx
  unsigned __int8 v47; // bl
  char v48; // al
  char *v49; // r13
  __int64 v50; // rax
  char v51; // al
  char *v52; // rbx
  __int64 j; // rax
  char v54; // al
  const char *v55; // r14
  char v56; // al
  const char *v57; // r15
  char v58; // al
  char *v59; // r15
  char *v60; // rax
  int v61; // r15d
  char *v62; // r14
  char v63; // al
  char v64; // al
  char *v65; // r15
  unsigned __int64 v66; // rax
  __int64 v67; // r13
  char *v68; // rbx
  __int64 k; // rax
  unsigned __int64 v71; // [rsp+8h] [rbp-68h]
  unsigned __int64 v72; // [rsp+10h] [rbp-60h]
  __int64 v74; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v75; // [rsp+28h] [rbp-48h]
  __int64 v76; // [rsp+30h] [rbp-40h]
  __int64 v77; // [rsp+38h] [rbp-38h]

  if ( unk_4F068C0 )
    qword_4CF7C90 = 0;
  v5 = sub_72FD90(*(_QWORD *)(a1 + 160), 10);
  v6 = v5;
  if ( v5 )
  {
    v71 = 0;
    v7 = 0;
    v72 = 0;
    while ( 1 )
    {
      v8 = sub_5D3810(v7, v6, a1);
      if ( v8 )
      {
        sub_5D5170(v7, v8);
        sub_5D45D0((unsigned int *)(v6 + 64));
        if ( *(char *)(v6 + 88) >= 0 )
        {
LABEL_7:
          if ( (*(_BYTE *)(v6 + 145) & 0x10) != 0 )
            goto LABEL_47;
          goto LABEL_8;
        }
      }
      else
      {
        sub_5D45D0((unsigned int *)(v6 + 64));
        if ( *(char *)(v6 + 88) >= 0 )
          goto LABEL_7;
      }
      v22 = 0;
      while ( 1 )
      {
        v23 = unk_4F04C50;
        if ( unk_4F04C50 )
          v23 = qword_4CF7E98;
        v24 = sub_732D20(v6, v23, 0, v22);
        v22 = v24;
        if ( !v24 )
          break;
        sub_5D52E0(v24, v23);
      }
      if ( (*(_BYTE *)(v6 + 145) & 0x10) != 0 )
      {
LABEL_47:
        for ( i = *(_QWORD *)(v6 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        if ( (*(_BYTE *)(v6 + 146) & 4) != 0 )
        {
          v50 = *(_QWORD *)(i + 168);
          if ( (*(_BYTE *)(v50 + 109) & 0x10) != 0 )
            i = *(_QWORD *)(v50 + 208);
        }
        v76 = v6;
        v75 = (_QWORD *)qword_4CF7CA8;
        if ( qword_4CF7CB0 )
          *(_QWORD *)qword_4CF7CA8 = &v74;
        else
          qword_4CF7CB0 = (__int64)&v74;
        qword_4CF7CA8 = (__int64)&v74;
        v77 = qword_4CF7CA0;
        qword_4CF7CA0 += *(_QWORD *)(v6 + 128);
        v74 = 0;
        sub_5DA0F0(i, a2, a3);
        v26 = sub_5D35E0(*a2);
        if ( *(_QWORD *)(*(_QWORD *)(v6 + 120) + 128LL) > v26 )
        {
          v27 = *(_QWORD *)(i + 128) - v26;
          if ( v27 )
            sub_5D5170(*a2, v27);
        }
        if ( unk_4F068C0 )
          qword_4CF7C90 = 0;
        qword_4CF7CA8 = (__int64)v75;
        if ( (__int64 *)qword_4CF7CB0 == &v74 )
          qword_4CF7CB0 = 0;
        else
          *v75 = 0;
        qword_4CF7CA0 = v77;
LABEL_19:
        if ( dword_4CF7EA0 )
        {
LABEL_20:
          if ( (*(_BYTE *)(v6 + 145) & 0x10) == 0 )
          {
            v31 = *(_QWORD *)(v6 + 128) + qword_4CF7CA0;
            putc(32, stream);
            v32 = dword_4CF7EA4;
            ++dword_4CF7F40;
            ++dword_4CF7EA4;
            if ( !v32 )
            {
              v56 = 47;
              v57 = "*";
              do
              {
                ++v57;
                putc(v56, stream);
                v56 = *(v57 - 1);
                ++dword_4CF7F40;
              }
              while ( v56 );
            }
            v33 = 32;
            v34 = "offset = ";
            do
            {
              ++v34;
              putc(v33, stream);
              v33 = *(v34 - 1);
            }
            while ( v33 );
            dword_4CF7F40 += 10;
            sub_5D32F0(v31);
            v35 = " bytes";
            v36 = (v31 != 1) + 5;
            if ( v31 == 1 )
              v35 = " byte";
            v37 = v35 + 1;
            v38 = *v35;
            do
            {
              ++v37;
              putc(v38, stream);
              v38 = *(v37 - 1);
            }
            while ( v38 );
            v39 = *(unsigned __int8 *)(v6 + 136);
            dword_4CF7F40 += v36;
            if ( v39 )
            {
              v58 = 44;
              v59 = " ";
              do
              {
                ++v59;
                putc(v58, stream);
                v58 = *(v59 - 1);
              }
              while ( v58 );
              dword_4CF7F40 += 2;
              sub_5D32F0(v39);
              v60 = " bits";
              v61 = (v39 != 1) + 4;
              if ( v39 == 1 )
                v60 = " bit";
              v62 = v60 + 1;
              v63 = *v60;
              do
              {
                ++v62;
                putc(v63, stream);
                v63 = *(v62 - 1);
              }
              while ( v63 );
              dword_4CF7F40 += v61;
            }
            v40 = 44;
            v41 = " type alignment = ";
            do
            {
              v42 = stream;
              ++v41;
              putc(v40, stream);
              v40 = *(v41 - 1);
            }
            while ( v40 );
            v44 = *(_QWORD *)(v6 + 120);
            dword_4CF7F40 += 19;
            if ( *(char *)(v44 + 142) >= 0 && *(_BYTE *)(v44 + 140) == 12 )
              v45 = (unsigned int)sub_8D4AB0(v44, v42, v43);
            else
              v45 = *(unsigned int *)(v44 + 136);
            sub_5D32F0(v45);
            putc(32, stream);
            ++dword_4CF7F40;
            if ( !--dword_4CF7EA4 )
            {
              v54 = 42;
              v55 = "/";
              do
              {
                ++v55;
                putc(v54, stream);
                v54 = *(v55 - 1);
                ++dword_4CF7F40;
              }
              while ( v54 );
            }
            putc(32, stream);
            ++dword_4CF7F40;
          }
        }
        if ( *(_BYTE *)(a1 + 140) != 11 )
          goto LABEL_22;
        goto LABEL_36;
      }
LABEL_8:
      if ( (*(_BYTE *)(v6 + 144) & 4) != 0 )
      {
        v9 = *(_QWORD *)(v6 + 184);
        if ( v9 )
        {
          if ( *(_BYTE *)(a1 + 140) == 11 )
          {
            *a3 = 1;
          }
          else
          {
            sub_74A390(*(_QWORD *)(v6 + 184), 0, 0, 0, 0, &qword_4CF7CE0);
            v28 = v9;
            v29 = " 0;";
            sub_74D110(v28, 0, 0, &qword_4CF7CE0);
            v30 = 58;
            do
            {
              ++v29;
              putc(v30, stream);
              v30 = *(v29 - 1);
            }
            while ( v30 );
            dword_4CF7F40 += 4;
            putc(32, stream);
            ++dword_4CF7F40;
          }
        }
        sub_5D4160(v6);
        if ( *(_QWORD *)(v6 + 8) )
        {
          putc(32, stream);
          v10 = *(_BYTE **)(v6 + 8);
          ++dword_4CF7F40;
          sub_5D4E40(v10, v6);
        }
        v11 = 58;
        v12 = " ";
        do
        {
          ++v12;
          putc(v11, stream);
          v11 = *(v12 - 1);
        }
        while ( v11 );
        v13 = *(unsigned __int8 *)(v6 + 137);
        dword_4CF7F40 += 2;
        sub_5D32F0(v13);
        sub_74FC40(v6, 1, &qword_4CF7CE0);
        putc(59, stream);
        v14 = *(unsigned __int8 *)(v6 + 137);
        v15 = *(_QWORD *)(v6 + 176);
        ++dword_4CF7F40;
        if ( v15 > v14 )
        {
          if ( *(_BYTE *)(a1 + 140) == 11 )
            *a3 = 1;
          else
            sub_5D3D20(*(_BYTE *)(v6 + 136), v14, v15);
        }
        goto LABEL_19;
      }
      v17 = *(_QWORD *)(v6 + 120);
      if ( unk_4F068E0
        && unk_4F068D8 <= 0x752Fu
        && (*(_BYTE *)(a1 + 179) & 8) != 0
        && (unsigned int)sub_8D3410(*(_QWORD *)(v6 + 120))
        && (unsigned int)sub_8D23B0(v17) )
      {
        for ( j = v17; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        *(_BYTE *)(j + 169) |= 0x20u;
      }
      if ( sub_5D7F20(v6) )
      {
        v18 = *(_QWORD *)(v17 + 8);
        *(_QWORD *)(v17 + 8) = byte_3F871B3;
        sub_5DAD30(v17, 0);
        *(_QWORD *)(v17 + 8) = v18;
      }
      else
      {
        sub_74A390(v17, 0, 1, 0, 1, &qword_4CF7CE0);
        sub_5D4E40(*(_BYTE **)(v6 + 8), v6);
        sub_74D110(v17, 0, 1, &qword_4CF7CE0);
      }
      sub_74FC40(v6, 1, &qword_4CF7CE0);
      putc(59, stream);
      ++dword_4CF7F40;
      if ( *(_BYTE *)(v17 + 140) == 12 )
      {
        v19 = v17;
        do
          v19 = *(_QWORD *)(v19 + 160);
        while ( *(_BYTE *)(v19 + 140) == 12 );
        if ( (*(_BYTE *)(v19 + 142) & 0x10) != 0 )
        {
          while ( 1 )
          {
            v17 = *(_QWORD *)(v17 + 160);
LABEL_33:
            if ( *(_BYTE *)(v17 + 140) != 12 )
              goto LABEL_34;
          }
        }
LABEL_96:
        if ( !(unsigned int)sub_8D3410(v17) )
          goto LABEL_99;
        for ( k = sub_8D40F0(v17); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
          ;
        if ( (*(_BYTE *)(k + 142) & 0x10) == 0 )
        {
LABEL_99:
          while ( *(_BYTE *)(v17 + 140) == 12 )
            v17 = *(_QWORD *)(v17 + 160);
          v71 = *(_QWORD *)(v17 + 128);
          goto LABEL_19;
        }
        goto LABEL_33;
      }
      if ( (*(_BYTE *)(v17 + 142) & 0x10) == 0 )
        goto LABEL_96;
LABEL_34:
      v20 = *(_QWORD *)(v17 + 128);
      if ( *(_BYTE *)(a1 + 140) != 11 )
      {
        v64 = 99;
        v65 = "har ";
        do
        {
          ++v65;
          putc(v64, stream);
          v64 = *(v65 - 1);
        }
        while ( v64 );
        dword_4CF7F40 += 5;
        ++dword_4CF7F60;
        sub_5D4E40("__dummy_empty", 0);
        v66 = sub_5D35E0(v6);
        sub_5D32F0(v66);
        --dword_4CF7F60;
        if ( v20 > 1 )
        {
          putc(91, stream);
          ++dword_4CF7F40;
          sub_5D32F0(v20);
          putc(93, stream);
          ++dword_4CF7F40;
        }
        putc(59, stream);
        ++dword_4CF7F40;
        goto LABEL_19;
      }
      v72 = v20;
      if ( dword_4CF7EA0 )
        goto LABEL_20;
LABEL_36:
      if ( *a2 )
      {
        v21 = sub_5D35E0(*a2);
        if ( v21 >= sub_5D35E0(v6) )
        {
          if ( !unk_4F068C0 )
            goto LABEL_23;
          goto LABEL_39;
        }
      }
LABEL_22:
      *a2 = v6;
      if ( !unk_4F068C0 )
        goto LABEL_23;
LABEL_39:
      sub_5D3540(v6);
LABEL_23:
      v7 = v6;
      v16 = sub_72FD90(*(_QWORD *)(v6 + 112), 10);
      if ( !v16 )
      {
        LOBYTE(v5) = v72 >= v71;
        if ( qword_4CF7CB0 && (*(_BYTE *)(v6 + 144) & 4) != 0 )
        {
          if ( unk_4F068C0 )
          {
            v67 = *(_QWORD *)(v6 + 120);
            v68 = "0;";
            sub_74A390(v67, 0, 0, 0, 0, &qword_4CF7CE0);
            sub_74D110(v67, 0, 0, &qword_4CF7CE0);
            LOBYTE(v5) = 58;
            do
            {
              ++v68;
              putc((char)v5, stream);
              LOBYTE(v5) = *(v68 - 1);
            }
            while ( (_BYTE)v5 );
            dword_4CF7F40 += 3;
          }
          else
          {
            v46 = *(unsigned __int8 *)(v6 + 137) + *(unsigned __int8 *)(v6 + 136);
            LODWORD(v5) = (dword_4F06BA0 - v46) / dword_4F06BA0;
            v47 = (dword_4F06BA0 - v46) % dword_4F06BA0;
            if ( v47 )
            {
              v48 = 117;
              v49 = "nsigned int:";
              do
              {
                ++v49;
                putc(v48, stream);
                v48 = *(v49 - 1);
              }
              while ( v48 );
              dword_4CF7F40 += 13;
              sub_5D32F0(v47);
              LOBYTE(v5) = putc(59, stream);
              ++dword_4CF7F40;
            }
          }
        }
        if ( v72 >= v71 && v72 != 0 )
        {
          v51 = 99;
          v52 = "har ";
          do
          {
            ++v52;
            putc(v51, stream);
            v51 = *(v52 - 1);
          }
          while ( v51 );
          dword_4CF7F40 += 5;
          sub_5D4E40("__dummy_empty", 0);
          if ( v72 != 1 )
          {
            putc(91, stream);
            ++dword_4CF7F40;
            sub_5D32F0(v72);
            putc(93, stream);
            ++dword_4CF7F40;
          }
          LOBYTE(v5) = putc(59, stream);
          ++dword_4CF7F40;
        }
        return v5;
      }
      v6 = v16;
    }
  }
  return v5;
}
