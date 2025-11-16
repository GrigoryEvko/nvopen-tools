// Function: sub_831CF0
// Address: 0x831cf0
//
__int64 __fastcall sub_831CF0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        int *a6,
        _DWORD *a7,
        int *a8,
        _DWORD *a9,
        _DWORD *a10,
        __int64 *a11)
{
  __int64 v11; // r14
  __int64 v12; // r12
  __int64 i; // r15
  __int64 j; // r9
  __int64 v15; // rsi
  unsigned int v16; // eax
  __int64 v17; // rcx
  __int64 v18; // r9
  unsigned int v19; // ebx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  int v23; // edx
  _BOOL4 v24; // r8d
  int v25; // eax
  char v26; // al
  __int64 v27; // r9
  _BOOL4 v28; // eax
  bool v29; // bl
  _BOOL4 v30; // eax
  unsigned int v31; // eax
  char v32; // al
  int v34; // eax
  int v35; // eax
  unsigned int v36; // eax
  _BOOL4 v37; // eax
  _BOOL4 v38; // eax
  _BOOL4 v39; // eax
  unsigned int v40; // eax
  int v41; // eax
  __int64 v42; // rax
  char v43; // cl
  int v44; // eax
  __int64 v45; // rax
  __int64 v46; // rax
  unsigned __int8 v47; // [rsp+Fh] [rbp-61h]
  __int64 v48; // [rsp+10h] [rbp-60h]
  __int64 v49; // [rsp+10h] [rbp-60h]
  __int64 v50; // [rsp+10h] [rbp-60h]
  __int64 v51; // [rsp+10h] [rbp-60h]
  unsigned int v52; // [rsp+10h] [rbp-60h]
  __int64 v53; // [rsp+10h] [rbp-60h]
  __int64 v54; // [rsp+10h] [rbp-60h]
  __int64 v55; // [rsp+10h] [rbp-60h]
  unsigned int v57; // [rsp+18h] [rbp-58h]
  unsigned int v58; // [rsp+1Ch] [rbp-54h]
  unsigned int v60; // [rsp+20h] [rbp-50h]
  int v61; // [rsp+20h] [rbp-50h]
  unsigned int v62; // [rsp+20h] [rbp-50h]
  unsigned int v63; // [rsp+20h] [rbp-50h]
  unsigned int v65; // [rsp+28h] [rbp-48h]
  unsigned int v66; // [rsp+28h] [rbp-48h]
  int v67; // [rsp+28h] [rbp-48h]
  unsigned int v68; // [rsp+28h] [rbp-48h]
  unsigned int v69; // [rsp+28h] [rbp-48h]
  unsigned int v70; // [rsp+28h] [rbp-48h]
  unsigned int v71; // [rsp+28h] [rbp-48h]
  unsigned int v72; // [rsp+28h] [rbp-48h]
  unsigned int v73; // [rsp+28h] [rbp-48h]
  int v74; // [rsp+34h] [rbp-3Ch] BYREF
  int v75; // [rsp+38h] [rbp-38h] BYREF
  unsigned int v76[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v11 = a2;
  if ( a11 )
    *a11 = 0;
  v12 = sub_8D46C0(a3);
  v58 = sub_8D3110(a3);
  if ( a1 )
  {
    if ( (unsigned int)sub_8D3410(v12) )
      sub_831BB0(a1, a2);
    v11 = *(_QWORD *)a1;
  }
  for ( i = v11; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  for ( j = v12; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
  {
    if ( (unsigned int)qword_4F077B4 | dword_4F077BC )
    {
      v15 = j;
      v54 = j;
      v36 = sub_8DED30(i, j, a4 == 0 ? 19 : 3);
      v18 = v54;
      v19 = v36;
    }
    else
    {
      v15 = v11;
      if ( v58 )
      {
        if ( a1 )
        {
          if ( *(_BYTE *)(a1 + 17) == 1 )
          {
            v50 = j;
            v30 = sub_6ED0A0(a1);
            v15 = i;
            j = v50;
            if ( v30 )
              v15 = v11;
          }
        }
      }
      v51 = j;
      v31 = sub_8E07E0(v12, v15);
      v18 = v51;
      v19 = v31;
    }
  }
  else
  {
    v15 = j;
    v48 = j;
    v16 = sub_8DF7B0(i, j, 0, 0, 0);
    v18 = v48;
    v19 = v16;
  }
  if ( v19 )
    goto LABEL_34;
  if ( dword_4F04C44 != -1
    || (v20 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v20 + 6) & 6) != 0)
    || *(_BYTE *)(v20 + 4) == 12 )
  {
    v53 = v18;
    if ( (unsigned int)sub_8DBE70(v18) || (unsigned int)sub_8DBE70(i) )
    {
      v21 = 1;
      v19 = 1;
LABEL_65:
      if ( (*(_BYTE *)(v12 + 140) & 0xFB) != 8 )
      {
        *a6 = 0;
        *a8 = 0;
        *a7 = 0;
        if ( v58 )
          goto LABEL_55;
        goto LABEL_67;
      }
      LOBYTE(v17) = 0;
      goto LABEL_52;
    }
    v18 = v53;
  }
  if ( (unsigned __int8)(*(_BYTE *)(v18 + 140) - 9) <= 2u && (unsigned __int8)(*(_BYTE *)(i + 140) - 9) <= 2u )
  {
    v15 = v18;
    v55 = v18;
    v45 = sub_8D5CE0(i, v18);
    v18 = v55;
    if ( v45 )
      goto LABEL_33;
    v17 = a4;
    if ( a4 )
    {
      v15 = i;
      v46 = sub_8D5CE0(v55, i);
      v18 = v55;
      if ( v46 )
        goto LABEL_33;
    }
  }
  if ( qword_4D0495C )
  {
    v49 = v18;
    if ( (unsigned int)sub_8D2E30(v18) )
    {
      if ( (unsigned int)sub_8D2E30(i) )
      {
        v15 = v49;
        if ( (unsigned int)sub_8DEFB0(i, v49, 0, 0) )
        {
LABEL_33:
          v19 = 1;
LABEL_34:
          if ( (*(_BYTE *)(v12 + 140) & 0xFB) != 8 )
          {
            v21 = 0;
            *a6 = 0;
            *a8 = 0;
            *a7 = 0;
            goto LABEL_36;
          }
          LOBYTE(v17) = 1;
          LODWORD(v21) = 0;
          goto LABEL_52;
        }
      }
    }
  }
  v21 = 0;
  if ( !a11 || !a1 || *(_WORD *)(a1 + 16) != 771 )
    goto LABEL_65;
  v22 = sub_82C9F0(
          *(_QWORD *)(a1 + 136),
          (*(_BYTE *)(a1 + 19) & 8) != 0,
          *(_QWORD *)(a1 + 104),
          1,
          a3,
          0,
          0,
          &v74,
          0,
          0,
          v76,
          &v75);
  v17 = (__int64)a11;
  v23 = v75;
  *a11 = v22;
  if ( v23 )
  {
    if ( (unsigned int)sub_6E5430() )
      sub_6854C0(0x1C1u, (FILE *)(a1 + 68), *(_QWORD *)(a1 + 136));
    sub_6E6840(a1);
    v21 = 0;
    v24 = 0;
    v17 = 0;
  }
  else
  {
    v21 = v76[0];
    if ( v76[0] )
    {
      v17 = 0;
      v21 = 1;
      v24 = 1;
    }
    else
    {
      LOBYTE(v17) = v22 != 0;
      v24 = v22 != 0;
    }
  }
  v19 = v24;
  if ( (*(_BYTE *)(v12 + 140) & 0xFB) != 8 )
  {
    v25 = 0;
    goto LABEL_53;
  }
LABEL_52:
  v47 = v17;
  v52 = v21;
  v32 = sub_8D4C10(v12, dword_4F077C4 != 2);
  v21 = v52;
  v17 = v47;
  v25 = v32 & 1;
LABEL_53:
  *a6 = v25;
  v15 = (__int64)a8;
  *a8 = v25;
  *a7 = 0;
  if ( !(_BYTE)v17 )
  {
LABEL_54:
    if ( v58 )
    {
LABEL_55:
      v27 = v19;
      *a8 = 1;
      if ( !a1 )
        goto LABEL_56;
      v70 = v21;
      v39 = sub_6ED230((_BYTE *)a1);
      v21 = v70;
      v27 = v19;
      if ( v39 )
        goto LABEL_56;
      goto LABEL_92;
    }
LABEL_67:
    v27 = v19;
    if ( !*a6 )
      goto LABEL_56;
    goto LABEL_68;
  }
LABEL_36:
  if ( !*a8 )
    goto LABEL_54;
  if ( a1 )
  {
    v26 = *(_BYTE *)(a1 + 16);
    if ( v26 == 1 )
    {
      v42 = *(_QWORD *)(a1 + 144);
      v27 = v19;
      v43 = *(_BYTE *)(v42 + 24);
      if ( v43 == 1 )
      {
        if ( *(_BYTE *)(v42 + 56) != 94 )
          goto LABEL_41;
        v42 = *(_QWORD *)(v42 + 72);
        if ( *(_BYTE *)(v42 + 24) != 3 )
          goto LABEL_41;
      }
      else if ( v43 != 3 )
      {
        goto LABEL_41;
      }
      v27 = 0;
      if ( (*(_BYTE *)(*(_QWORD *)(v42 + 56) - 8LL) & 0x10) == 0 )
        v27 = v19;
    }
    else
    {
      v27 = v19;
      if ( v26 == 2 && *(_BYTE *)(a1 + 317) == 6 && !*(_QWORD *)(a1 + 336) && *(_BYTE *)(a1 + 320) == 1 )
      {
        v27 = 0;
        if ( (*(_BYTE *)(*(_QWORD *)(a1 + 328) - 8LL) & 0x10) == 0 )
          v27 = v19;
      }
    }
LABEL_41:
    v17 = v58;
    if ( v58 )
    {
      v60 = v27;
      v65 = v21;
      *a8 = 1;
      v28 = sub_6ED230((_BYTE *)a1);
      v21 = v65;
      v27 = v60;
      if ( v28 )
      {
        v15 = (unsigned int)*a8;
        v29 = (_DWORD)v15 == 0;
LABEL_73:
        if ( v29 )
        {
          if ( *(_BYTE *)(a1 + 17) == 2 || (v62 = v27, v68 = v21, v38 = sub_6ED0A0(a1), v21 = v68, v27 = v62, v38) )
          {
            if ( (_DWORD)v21 )
            {
              if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
              {
                v63 = v27;
                v69 = v21;
                sub_8D3A70(i);
                v21 = v69;
                v27 = v63;
              }
              *a9 = 0;
              goto LABEL_59;
            }
            v73 = v27;
            v44 = sub_8D3A70(i);
            v27 = v73;
            LODWORD(v21) = v44;
            if ( !v44 )
            {
              if ( dword_4F077BC )
              {
                if ( dword_4F04C44 != -1
                  || (v17 = (__int64)qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0) )
                {
                  v21 = 1;
                  *a9 = 0;
                  goto LABEL_59;
                }
              }
              LODWORD(v27) = 0;
              *a9 = 0;
              if ( (*(_BYTE *)(v11 + 140) & 0xFB) != 8 )
                goto LABEL_62;
              goto LABEL_105;
            }
            *a9 = 0;
LABEL_75:
            if ( (*(_BYTE *)(v11 + 140) & 0xFB) != 8 )
            {
LABEL_76:
              v29 = *a9 == 0;
              v21 = 0;
              goto LABEL_59;
            }
LABEL_105:
            v72 = v27;
            v15 = dword_4F077C4 != 2;
            v41 = sub_8D4C10(v11, v15);
            v27 = v72;
            if ( v41 )
            {
              if ( (*(_BYTE *)(v12 + 140) & 0xFB) != 8
                || (v15 = dword_4F077C4 != 2, (v41 & ~(unsigned int)sub_8D4C10(v12, v15)) != 0) )
              {
                LODWORD(v21) = 0;
                LODWORD(v27) = 0;
                *a9 = 1;
                goto LABEL_62;
              }
              v27 = v72;
              v29 = *a9 == 0;
              v21 = 0;
              goto LABEL_59;
            }
            goto LABEL_76;
          }
        }
        goto LABEL_74;
      }
LABEL_92:
      if ( (_DWORD)v27 )
      {
        v17 = a4;
        if ( !a4 )
        {
          if ( dword_4F077BC )
          {
            if ( qword_4F077A8 >= 0x9E34u )
              v27 = a4;
          }
          else
          {
            v27 = 0;
          }
        }
      }
      goto LABEL_56;
    }
    goto LABEL_99;
  }
  v27 = v19;
  if ( v58 )
  {
    v27 = v19;
    *a8 = 1;
LABEL_74:
    *a9 = 0;
    if ( !(_DWORD)v21 )
      goto LABEL_75;
    goto LABEL_58;
  }
LABEL_99:
  if ( !*a6 )
    goto LABEL_74;
LABEL_68:
  v57 = v27;
  v66 = v21;
  v34 = sub_8D3070(a3);
  v21 = v66;
  v27 = v57;
  if ( v34 )
  {
    v35 = sub_8D4D20(a3);
    v21 = v66;
    v27 = v57;
    if ( !v35 )
    {
      *a8 = 0;
      *a7 = 1;
    }
  }
LABEL_56:
  if ( v19 )
  {
    v29 = a1 != 0 && *a8 == 0;
    goto LABEL_73;
  }
  *a9 = 0;
  if ( !(_DWORD)v21 )
  {
    v15 = v11;
    v71 = v27;
    v40 = sub_8DF8D0(v12, v11);
    v27 = v71;
    v21 = v40;
    if ( !v40 )
    {
      v29 = *a9 == 0;
      goto LABEL_59;
    }
    goto LABEL_75;
  }
LABEL_58:
  v29 = 1;
LABEL_59:
  if ( (_DWORD)v27 && v29 )
  {
    if ( *a8 )
    {
      if ( !v58 )
      {
        if ( a1 )
        {
          v61 = v27;
          v67 = v21;
          v37 = sub_6ECD10(a1, v15, v21, v17, v58, v27);
          LODWORD(v21) = v67;
          LODWORD(v27) = v61;
          if ( (v67 & 1) == 0 && v37 )
          {
            LODWORD(v21) = 0;
            LODWORD(v27) = 0;
          }
        }
      }
    }
  }
  else
  {
    LODWORD(v27) = 0;
  }
LABEL_62:
  *a10 = v21;
  return (unsigned int)v27;
}
