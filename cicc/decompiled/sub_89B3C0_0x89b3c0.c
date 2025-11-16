// Function: sub_89B3C0
// Address: 0x89b3c0
//
__int64 __fastcall sub_89B3C0(__int64 a1, __int64 a2, int a3, unsigned int a4, _DWORD *a5, unsigned __int8 a6)
{
  int v6; // r10d
  __int64 v8; // r12
  int v9; // r9d
  __int64 v10; // rax
  int v11; // ebx
  __int64 v12; // rax
  __int64 v14; // rbx
  __int64 v15; // r11
  __int64 v16; // r8
  unsigned __int8 v17; // dl
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // r13
  bool v21; // r15
  char v22; // al
  __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rdi
  unsigned int v26; // edx
  int v27; // eax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rdi
  __int64 v31; // rsi
  bool v32; // dl
  bool v33; // cl
  bool v34; // di
  int v35; // eax
  __int64 v36; // rdi
  __int64 v37; // rsi
  _BOOL4 v38; // eax
  __int64 v39; // rcx
  __int128 v40; // rax
  _BOOL4 v41; // eax
  _BOOL4 v42; // eax
  int v43; // eax
  __int64 v44; // rdi
  int v45; // eax
  int v46; // eax
  int v47; // [rsp+4h] [rbp-6Ch]
  unsigned int v48; // [rsp+4h] [rbp-6Ch]
  int v49; // [rsp+4h] [rbp-6Ch]
  int v50; // [rsp+4h] [rbp-6Ch]
  int v51; // [rsp+4h] [rbp-6Ch]
  int v54; // [rsp+14h] [rbp-5Ch]
  bool v56; // [rsp+23h] [rbp-4Dh]
  unsigned __int8 v57; // [rsp+24h] [rbp-4Ch]
  int v58; // [rsp+28h] [rbp-48h]
  __int64 v59; // [rsp+28h] [rbp-48h]
  int v60; // [rsp+28h] [rbp-48h]
  __int64 v61; // [rsp+28h] [rbp-48h]
  __int64 v62; // [rsp+28h] [rbp-48h]
  __int64 v63; // [rsp+28h] [rbp-48h]
  __int64 v65; // [rsp+30h] [rbp-40h]
  unsigned int v66; // [rsp+30h] [rbp-40h]
  __int64 v67; // [rsp+30h] [rbp-40h]
  unsigned int v68; // [rsp+30h] [rbp-40h]
  unsigned int v69; // [rsp+30h] [rbp-40h]
  unsigned int v70; // [rsp+30h] [rbp-40h]
  int v72; // [rsp+3Ch] [rbp-34h]
  int v73; // [rsp+3Ch] [rbp-34h]
  int v74; // [rsp+3Ch] [rbp-34h]
  int v76; // [rsp+3Ch] [rbp-34h]
  int v77; // [rsp+3Ch] [rbp-34h]
  int v78; // [rsp+3Ch] [rbp-34h]

  v6 = a3;
  v8 = a2;
  v9 = a4 & 4;
  v56 = v9 != 0;
  v54 = a4 & 2;
  if ( (a4 & 2) != 0 )
  {
    if ( !a2 || !a1 )
    {
      v14 = a1;
      v18 = a2;
      v32 = a1 != 0;
      v15 = 0;
      v8 = 0;
      LODWORD(v16) = 0;
      v33 = a1 != 0 && v56;
      goto LABEL_39;
    }
  }
  else
  {
    v72 = v9;
    if ( !a1 )
    {
      if ( !a2 )
      {
        LODWORD(v16) = 0;
        return (unsigned int)v16 ^ 1;
      }
      v14 = 0;
      sub_892BC0(a2);
      v6 = a3;
      v18 = a2;
      v15 = 0;
      LODWORD(v16) = 0;
      v8 = 0;
      v32 = 0;
      goto LABEL_44;
    }
    v10 = sub_892BC0(a1);
    v6 = a3;
    if ( !a2 )
    {
      v33 = v56;
      v14 = a1;
      v15 = 0;
      LODWORD(v16) = 0;
      v18 = 0;
      v32 = 1;
      goto LABEL_39;
    }
    v11 = *(_DWORD *)(v10 + 4);
    v12 = sub_892BC0(a2);
    v6 = a3;
    v9 = v72;
    if ( *(_DWORD *)(v12 + 4) != 0 && v11 != *(_DWORD *)(v12 + 4) && v11 )
      return 0;
  }
  v14 = a1;
  v15 = 0;
  v16 = 0;
  v57 = a6;
  while ( 1 )
  {
    v19 = *(_QWORD *)(v14 + 8);
    v20 = *(_QWORD *)(v8 + 8);
    v21 = v9 == 0;
    v22 = *(_BYTE *)(v19 + 80);
    if ( v22 != *(_BYTE *)(v20 + 80)
      || (a4 & 0x10) != 0 && ((*(_BYTE *)(v14 + 57) & 2) != 0 || (*(_BYTE *)(v8 + 57) & 2) != 0) )
    {
      goto LABEL_10;
    }
    v17 = *(_BYTE *)(v14 + 56);
    v23 = (v17 ^ *(_BYTE *)(v8 + 56)) & 0x10;
    if ( ((v17 ^ *(_BYTE *)(v8 + 56)) & 0x10) != 0 && ((v17 & 0x10) == 0 || !v9) )
    {
      v16 = 1;
      if ( !v6 )
        goto LABEL_14;
LABEL_26:
      v58 = v6;
      v65 = v15;
      v73 = v9;
      sub_6853B0(v57, 0x93u, (FILE *)(v20 + 48), v19);
      v17 = *(_BYTE *)(v14 + 56);
      v6 = v58;
      v16 = 1;
      v15 = v65;
      v9 = v73;
      goto LABEL_12;
    }
    v24 = *(_QWORD *)(v8 + 64);
    v25 = *(_QWORD *)(v14 + 64);
    if ( v22 == 3 )
    {
      if ( *(_BYTE *)(v24 + 140) != 14 || *(_BYTE *)(v25 + 140) != 14 )
        goto LABEL_12;
      *(_QWORD *)&v40 = *(_QWORD *)(*(_QWORD *)(v24 + 168) + 32LL);
      *((_QWORD *)&v40 + 1) = *(_QWORD *)(*(_QWORD *)(v25 + 168) + 32LL);
      if ( (_QWORD)v40 && *((_QWORD *)&v40 + 1) )
      {
        if ( *(_QWORD *)(v40 + 56) != *(_QWORD *)(*((_QWORD *)&v40 + 1) + 56LL) )
          goto LABEL_10;
        v49 = v6;
        v61 = v15;
        v68 = v16;
        v77 = v9;
        v41 = sub_89AB40(
                *(_QWORD *)(v40 + 64),
                *(_QWORD *)(*((_QWORD *)&v40 + 1) + 64LL),
                2,
                v23,
                (_UNKNOWN *__ptr32 *)v16);
        v9 = v77;
        v16 = v68;
        v15 = v61;
        v6 = v49;
        v42 = !v41;
      }
      else
      {
        v42 = v40 != 0;
      }
      if ( !v42 )
      {
        if ( (a4 & 8) == 0 )
          goto LABEL_37;
        v30 = *(_QWORD *)(v14 + 80);
        v31 = *(_QWORD *)(v8 + 80);
        if ( v30 && v31 )
        {
          if ( v30 == v31 )
            goto LABEL_37;
          v50 = v6;
          v62 = v15;
          v69 = v16;
          v78 = v9;
          v43 = sub_8D97D0(v30, v31, 0, v23, v16);
          v9 = v78;
          v16 = v69;
          v15 = v62;
          v6 = v50;
          if ( v43 )
          {
LABEL_37:
            v17 = *(_BYTE *)(v14 + 56);
            goto LABEL_12;
          }
          goto LABEL_10;
        }
LABEL_36:
        if ( v30 == v31 )
          goto LABEL_37;
      }
    }
    else if ( v22 == 2 )
    {
      if ( (*(_BYTE *)(v14 + 56) & 0x10) == 0 || (v26 = 8, !v56) )
        v26 = 8 * (v54 != 0);
      v47 = v6;
      v59 = v15;
      v66 = v16;
      v74 = v9;
      v27 = sub_739430(v25, v24, v26, v23, (_UNKNOWN *__ptr32 *)v16);
      v9 = v74;
      v16 = v66;
      v15 = v59;
      v6 = v47;
      if ( v27 )
        goto LABEL_33;
      if ( (a4 & 1) != 0 )
      {
        v44 = *(_QWORD *)(v14 + 64);
        if ( *(_BYTE *)(v44 + 173) == 12 && !*(_BYTE *)(v44 + 176) )
        {
          v45 = sub_88F430(v44, *(__m128i **)(v8 + 64));
          v16 = v66;
          v9 = v74;
          v15 = v59;
          v6 = v47;
          v28 = v66 | (v45 == 0);
          if ( !(v66 | (v45 == 0)) )
          {
            sub_685490(0x93u, (FILE *)(v20 + 48), v19);
            v9 = v74;
            v16 = v66;
            v15 = v59;
            v6 = v47;
LABEL_33:
            if ( (a4 & 8) == 0 )
              goto LABEL_37;
            v30 = *(_QWORD *)(v14 + 80);
            v31 = *(_QWORD *)(v8 + 80);
            if ( !v30 || !v31 )
              goto LABEL_36;
            v51 = v6;
            v63 = v15;
            v70 = v16;
            v76 = v9;
            v46 = sub_73A2C0(v30, v31, v28, v29, (_UNKNOWN *__ptr32 *)v16);
            goto LABEL_91;
          }
          if ( v45 )
            goto LABEL_33;
        }
      }
    }
    else
    {
      v60 = v6;
      v67 = v15;
      v76 = v9;
      v48 = v16;
      v35 = sub_89B9E0(v25, v24, (v17 >> 3) & 2, a4);
      v9 = v76;
      v15 = v67;
      v6 = v60;
      if ( v35 )
      {
        v16 = v48;
        if ( (a4 & 8) == 0 )
          goto LABEL_37;
        v36 = *(_QWORD *)(v14 + 80);
        v37 = *(_QWORD *)(v8 + 80);
        if ( !v36 || !v37 )
        {
          v38 = v36 != v37;
          goto LABEL_53;
        }
        v51 = v60;
        v63 = v67;
        v70 = v16;
        v46 = sub_89AAB0(v36, v37, 0);
LABEL_91:
        v9 = v76;
        v16 = v70;
        v15 = v63;
        v6 = v51;
        v38 = v46 == 0;
LABEL_53:
        if ( !v38 )
        {
          v17 = *(_BYTE *)(v14 + 56);
          goto LABEL_12;
        }
      }
    }
LABEL_10:
    if ( v6 )
      goto LABEL_26;
    v17 = *(_BYTE *)(v14 + 56);
    v16 = 1;
LABEL_12:
    if ( (v17 & 0x10) == 0 || v21 )
    {
LABEL_14:
      v15 = v14;
      v14 = *(_QWORD *)v14;
    }
    v18 = *(_QWORD *)v8;
    if ( !*(_QWORD *)v8 || !v14 )
      break;
    v8 = *(_QWORD *)v8;
  }
  v32 = v14 != 0;
  v33 = v14 != 0 && v56;
LABEL_39:
  if ( v33 && (*(_BYTE *)(v14 + 56) & 0x10) != 0 )
    return (unsigned int)v16 ^ 1;
LABEL_44:
  v34 = v18 != 0 && v56;
  if ( unk_4D04854 )
  {
    if ( v34 )
    {
      if ( (*(_BYTE *)(v18 + 56) & 0x10) != 0 )
        return (unsigned int)v16 ^ 1;
      goto LABEL_56;
    }
LABEL_67:
    if ( !v32 && !v18 )
      return (unsigned int)v16 ^ 1;
  }
  else
  {
    if ( !v34 )
      goto LABEL_67;
LABEL_56:
    if ( HIDWORD(qword_4F077B4) && qword_4F077A8 <= 0x9D07u && (*(_BYTE *)(v18 + 56) & 1) != 0 )
      return (unsigned int)v16 ^ 1;
  }
  if ( !v6 )
    return 0;
  if ( v14 )
  {
    if ( v8 )
      a5 = (_DWORD *)(*(_QWORD *)(v8 + 8) + 48LL);
    if ( v15 )
      v39 = *(_QWORD *)(v15 + 8);
    else
      v39 = *(_QWORD *)(a1 + 8);
    sub_6854F0(a6, 0x1F9u, a5, (_QWORD *)(v39 + 48));
  }
  else
  {
    sub_6854F0(a6, 0x1FAu, (_DWORD *)(*(_QWORD *)(v18 + 8) + 48LL), (_QWORD *)(*(_QWORD *)(v15 + 8) + 48LL));
  }
  return 0;
}
