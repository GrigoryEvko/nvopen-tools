// Function: sub_846560
// Address: 0x846560
//
__int64 *__fastcall sub_846560(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        _BYTE *a4,
        __int64 a5,
        _BOOL4 a6,
        _BOOL4 *a7,
        __int64 *a8,
        __int64 *a9)
{
  __int64 v9; // r11
  _BOOL4 v10; // r15d
  __m128i *v12; // r13
  __int64 v13; // r12
  bool v14; // zf
  _QWORD *v15; // rcx
  __int64 v16; // rdi
  char v17; // al
  bool v18; // bl
  int v19; // eax
  int v20; // eax
  int v21; // eax
  const __m128i *v22; // rdi
  __int64 v23; // rcx
  const __m128i *v24; // rax
  int v25; // r10d
  __int64 v26; // rax
  __int64 v27; // rax
  _BOOL4 v28; // eax
  int v29; // eax
  int v30; // eax
  __int64 v31; // rdx
  int v33; // eax
  int v34; // eax
  char v35; // al
  const __m128i *v36; // rdi
  __int64 v37; // rcx
  __int64 v38; // r8
  int v39; // eax
  int v40; // eax
  char v41; // al
  __m128i *v42; // rdi
  __int64 *v43; // r14
  __int64 v44; // rax
  const __m128i *v45; // rdx
  __m128i *v46; // rax
  int v47; // eax
  int v48; // r10d
  _BOOL4 v49; // edx
  _BOOL4 v50; // eax
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // rax
  __int64 v57; // rax
  int v58; // r8d
  int v59; // r10d
  unsigned __int8 v60; // di
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rcx
  int v66; // r10d
  __int64 v67; // r12
  const __m128i *v68; // rdi
  __int64 v69; // rax
  __int64 v70; // rax
  __m128i *v71; // rax
  __int64 v72; // rax
  unsigned int v73; // eax
  __int64 v74; // [rsp+8h] [rbp-78h]
  __int64 v75; // [rsp+8h] [rbp-78h]
  __int64 v76; // [rsp+8h] [rbp-78h]
  __int64 v77; // [rsp+10h] [rbp-70h]
  __int64 v78; // [rsp+10h] [rbp-70h]
  __int64 v79; // [rsp+10h] [rbp-70h]
  __int64 v80; // [rsp+10h] [rbp-70h]
  __int64 v81; // [rsp+10h] [rbp-70h]
  int v82; // [rsp+10h] [rbp-70h]
  __int64 v83; // [rsp+10h] [rbp-70h]
  __int64 v84; // [rsp+10h] [rbp-70h]
  __int64 v85; // [rsp+10h] [rbp-70h]
  int v86; // [rsp+10h] [rbp-70h]
  __int64 v87; // [rsp+10h] [rbp-70h]
  __int64 v88; // [rsp+10h] [rbp-70h]
  __int64 v89; // [rsp+10h] [rbp-70h]
  int v90; // [rsp+10h] [rbp-70h]
  _BOOL4 v91; // [rsp+10h] [rbp-70h]
  int v93; // [rsp+24h] [rbp-5Ch]
  int v94; // [rsp+24h] [rbp-5Ch]
  __int64 v95; // [rsp+28h] [rbp-58h]
  int v96; // [rsp+28h] [rbp-58h]
  int v97; // [rsp+28h] [rbp-58h]
  int v98; // [rsp+28h] [rbp-58h]
  unsigned __int8 v99; // [rsp+28h] [rbp-58h]
  int v100; // [rsp+28h] [rbp-58h]
  int v101; // [rsp+28h] [rbp-58h]
  unsigned int v102; // [rsp+34h] [rbp-4Ch] BYREF
  __int64 v103; // [rsp+38h] [rbp-48h] BYREF
  __int64 v104; // [rsp+40h] [rbp-40h] BYREF
  __int64 v105[7]; // [rsp+48h] [rbp-38h] BYREF

  v9 = a3;
  v10 = a6;
  v12 = (__m128i *)a2;
  v13 = a2;
  v14 = *(_BYTE *)(a2 + 140) == 12;
  v93 = a5;
  v103 = 0;
  if ( v14 )
  {
    do
      v13 = *(_QWORD *)(v13 + 160);
    while ( *(_BYTE *)(v13 + 140) == 12 );
  }
  v15 = qword_4F04C68;
  v105[0] = 0;
  v16 = *(_QWORD *)a3;
  v95 = *(_QWORD *)a3;
  v17 = *(_BYTE *)(a3 + 16);
  v18 = (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) & 0x40) != 0;
  if ( (v17 & 1) != 0 )
    goto LABEL_34;
  if ( !v16 )
  {
    v102 = 0;
    if ( (*(_BYTE *)(a3 + 17) & 1) != 0 )
    {
      v10 = 0;
    }
    else if ( (v17 & 2) != 0 )
    {
      v102 = 1;
      v36 = (const __m128i *)*a1;
      if ( a6 )
        goto LABEL_95;
      v10 = 1;
    }
    else if ( a6 )
    {
      goto LABEL_91;
    }
LABEL_18:
    if ( *((_BYTE *)a1 + 16) )
    {
      v22 = (const __m128i *)*a1;
      goto LABEL_20;
    }
    v31 = v103;
LABEL_134:
    a5 = v31;
LABEL_27:
    if ( !a5 )
    {
      if ( !a9 )
      {
LABEL_47:
        v31 = v103;
        goto LABEL_48;
      }
LABEL_29:
      v26 = v105[0];
      if ( !v105[0] )
      {
        v27 = (__int64)sub_7305B0();
LABEL_46:
        *a9 = v27;
        goto LABEL_47;
      }
LABEL_45:
      *(_BYTE *)(v26 + 25) |= 1u;
      *(_QWORD *)v26 = v12;
      sub_6EB510(v105[0]);
      v27 = v105[0];
      goto LABEL_46;
    }
    goto LABEL_39;
  }
  if ( (*(_BYTE *)(v16 + 194) & 4) != 0 && (v17 & 2) != 0 && *((_BYTE *)a1 + 16) != 2 )
  {
    v79 = a3;
    v28 = sub_72F570(v16);
    v9 = v79;
    if ( !v28 )
    {
      if ( *a1 == a2 || (v29 = sub_8D97D0(*a1, a2, 32, v15, a5), v9 = v79, v29) )
      {
LABEL_34:
        v14 = *((_BYTE *)a1 + 16) == 5;
        v102 = 1;
        if ( v14 )
        {
          v83 = v9;
          sub_8422F0((const __m128i *)a1, (__m128i *)a2);
          v9 = v83;
          if ( dword_4F077C4 != 2 )
            goto LABEL_36;
        }
        else if ( dword_4F077C4 != 2 )
        {
LABEL_36:
          v30 = 0;
          goto LABEL_60;
        }
        if ( *a1 != v13 )
        {
          v80 = v9;
          v33 = sub_8D97D0(*a1, v13, 32, v15, a5);
          v9 = v80;
          if ( !v33 )
            goto LABEL_36;
        }
        v81 = v9;
        a2 = v93 == 0;
        v34 = sub_831A40((__int64)a1, a2, v105, &v103);
        v9 = v81;
        if ( !v34 )
        {
          v30 = 1;
LABEL_60:
          a2 = (__int64)a1 + 68;
          v74 = v9;
          v82 = v30;
          sub_6E61E0(v13, (__int64)a1 + 68, 0);
          v9 = v74;
          if ( v10 )
          {
            if ( v82 )
            {
              v10 = 0;
              sub_82AFD0(v13, (__int64)a1 + 68);
              a5 = v103;
              v9 = v74;
              a2 = 1;
              if ( v103 )
                goto LABEL_40;
            }
            else
            {
              a5 = v103;
              v10 = 0;
              a2 = 1;
              if ( v103 )
                goto LABEL_40;
            }
            goto LABEL_18;
          }
          goto LABEL_37;
        }
        v102 = 0;
        v36 = (const __m128i *)*a1;
        if ( v10 )
          goto LABEL_95;
        v10 = 1;
LABEL_37:
        a3 = v103;
        a5 = v103;
        goto LABEL_38;
      }
    }
  }
  v102 = 0;
  if ( (*(_BYTE *)(v9 + 17) & 1) != 0 )
  {
    a3 = v103;
    a5 = v103;
    if ( !v10 )
      goto LABEL_38;
    a2 = 1;
    v10 = 0;
    goto LABEL_18;
  }
  if ( *(_BYTE *)(v95 + 174) == 1 )
  {
    if ( *a1 == v13 || (a2 = v13, v77 = v9, v19 = sub_8D97D0(*a1, v13, 32, v15, a5), v9 = v77, v19) )
    {
      a2 = v13;
      v78 = v9;
      v20 = sub_72F500(v95, v13, 0, 1, 0);
      v9 = v78;
      if ( v20 )
      {
        if ( (*(_BYTE *)(v78 + 16) & 2) != 0 && *((_BYTE *)a1 + 16) == 2 )
        {
          v102 = 1;
        }
        else
        {
          v102 = 0;
          a2 = v93 == 0;
          v21 = sub_831A40((__int64)a1, a2, v105, &v103);
          v9 = v78;
          if ( !v21 )
          {
            a3 = v103;
            a5 = v103;
            if ( v10 )
            {
              v10 = 0;
              a2 = 1;
              if ( v103 )
              {
LABEL_40:
                if ( *(_BYTE *)(a5 + 48) == 5 && (_BYTE)a2 )
                  *(_BYTE *)(a5 + 72) |= 2u;
                if ( !a9 )
                  goto LABEL_47;
                v26 = v105[0];
                if ( !v105[0] )
                {
                  v27 = sub_6EC670((__int64)v12, a5, 1, 0);
                  goto LABEL_46;
                }
                goto LABEL_45;
              }
              goto LABEL_18;
            }
LABEL_38:
            if ( a3 )
              goto LABEL_39;
            goto LABEL_18;
          }
        }
        if ( v10 )
        {
          v36 = (const __m128i *)*a1;
          a2 = v95;
          goto LABEL_96;
        }
LABEL_122:
        a3 = v103;
        v10 = 1;
        a5 = v103;
        goto LABEL_38;
      }
    }
    v35 = *(_BYTE *)(v9 + 16) & 0x40;
    if ( !v10 )
    {
      v10 = v35 == 0;
      goto LABEL_37;
    }
    v36 = (const __m128i *)v13;
    if ( v35 )
    {
      a5 = v103;
LABEL_91:
      v25 = 0;
LABEL_92:
      if ( !*((_BYTE *)a1 + 16) )
      {
        a5 = v103;
        v10 = 0;
        goto LABEL_25;
      }
      v24 = (const __m128i *)*a1;
      v10 = 0;
      v23 = *(unsigned __int8 *)(*a1 + 140LL);
      goto LABEL_22;
    }
LABEL_95:
    a2 = 0;
    goto LABEL_96;
  }
  v84 = v9;
  sub_8449E0(a1, 0, v9, 0, 0);
  v9 = v84;
  if ( *a1 == v13 || (v39 = sub_8D97D0(*a1, v13, 32, v37, v38), v9 = v84, v39) )
  {
    v85 = v9;
    a2 = v93 == 0;
    v40 = sub_831A40((__int64)a1, a2, v105, &v103);
    v9 = v85;
    if ( v40 )
    {
      if ( v10 )
      {
        v36 = (const __m128i *)*a1;
        a2 = 0;
LABEL_96:
        v87 = v9;
        v10 = 1;
        sub_83EB20(v36, (__int64 *)a2, (FILE *)((char *)a1 + 68));
        a3 = v103;
        v9 = v87;
        a5 = v103;
        goto LABEL_38;
      }
      goto LABEL_122;
    }
  }
  if ( !*((_BYTE *)a1 + 16) )
  {
    v31 = v103;
    a5 = v103;
    if ( v10 )
    {
      LOBYTE(a2) = 1;
      v10 = 0;
      if ( v103 )
        goto LABEL_40;
      if ( a9 )
        goto LABEL_29;
      goto LABEL_48;
    }
    if ( v103 )
      goto LABEL_39;
    goto LABEL_134;
  }
  v22 = (const __m128i *)*a1;
  v23 = *(unsigned __int8 *)(*a1 + 140LL);
  v24 = (const __m128i *)*a1;
  if ( (_BYTE)v23 == 12 )
  {
    v45 = (const __m128i *)*a1;
    do
    {
      v45 = (const __m128i *)v45[10].m128i_i64[0];
      a2 = v45[8].m128i_u8[12];
    }
    while ( (_BYTE)a2 == 12 );
  }
  else
  {
    a2 = *(unsigned __int8 *)(*a1 + 140LL);
  }
  if ( (_BYTE)a2 )
  {
    if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400100) == 0
      && dword_4F077BC
      && (dword_4F04C44 != -1
       || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0
       || unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0)
      && (v88 = v9, v46 = sub_73D7F0(*(_QWORD *)(v95 + 152)), v47 = sub_8D32E0(v46), v9 = v88, v47) )
    {
      v48 = 1;
      *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) |= 0x40u;
    }
    else
    {
      v48 = 0;
    }
    v49 = 1;
    if ( *((_BYTE *)a1 + 17) != 2 )
    {
      v89 = v9;
      v96 = v48;
      v50 = sub_6ED0A0((__int64)a1);
      v9 = v89;
      v48 = v96;
      v49 = v50;
    }
    a2 = 0;
    if ( (*(_BYTE *)(*a1 + 140LL) & 0xFB) == 8 )
    {
      v76 = v9;
      v91 = v49;
      v101 = v48;
      v73 = sub_8D4C10(*a1, dword_4F077C4 != 2);
      v9 = v76;
      v49 = v91;
      v48 = v101;
      a2 = v73;
    }
    v75 = v9;
    v10 = 0;
    v90 = v48;
    v51 = sub_6EB190(v13, a2, v49, (int)a1 + 68, (int)&v102, 0);
    a5 = v103;
    v25 = v90;
    v95 = v51;
    v9 = v75;
    if ( v103 )
    {
LABEL_87:
      if ( v25 )
      {
LABEL_26:
        *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) = (v18 << 6)
                                                                  | *(_BYTE *)(qword_4F04C68[0]
                                                                             + 776LL * dword_4F04C64
                                                                             + 13)
                                                                  & 0xBF;
        goto LABEL_27;
      }
LABEL_39:
      LOBYTE(a2) = !v10;
      goto LABEL_40;
    }
    goto LABEL_92;
  }
  a3 = v103;
  a5 = v103;
  if ( v10 )
  {
    a2 = 1;
    v10 = 0;
    if ( v103 )
      goto LABEL_40;
    v95 = 0;
    v25 = 0;
    goto LABEL_22;
  }
  if ( v103 )
    goto LABEL_39;
  v95 = 0;
LABEL_20:
  v23 = v22[8].m128i_u8[12];
  v24 = v22;
  v25 = 0;
LABEL_22:
  while ( (_BYTE)v23 == 12 )
  {
    v24 = (const __m128i *)v24[10].m128i_i64[0];
    v23 = v24[8].m128i_u8[12];
  }
  if ( !(_BYTE)v23 )
  {
    a5 = v103;
    goto LABEL_25;
  }
  if ( v102 )
  {
    v98 = v25;
    sub_8424A0((__m128i *)a1, v12, a3, v23, a5);
    v58 = v93;
    v59 = v98;
    v99 = *((_BYTE *)a1 + 16);
    v94 = v59;
    v60 = (v99 != 2) + 2;
    if ( v58 )
    {
      v61 = sub_6EB460(v60, v13, (_QWORD *)((char *)a1 + 68));
      v66 = v94;
      v65 = v99;
    }
    else
    {
      v61 = sub_6EAFA0(v60);
      v65 = v99;
      v66 = v94;
    }
    v67 = v61;
    v103 = v61;
    v68 = (const __m128i *)(a1 + 18);
    v100 = v66;
    if ( (_BYTE)v65 != 2 )
    {
      v69 = sub_6F6F40((const __m128i *)a1, 0, v62, v65, v63, v64);
      a5 = v103;
      v25 = v100;
      *(_QWORD *)(v67 + 56) = v69;
      goto LABEL_25;
    }
    goto LABEL_142;
  }
  v41 = *(_BYTE *)(v9 + 17);
  if ( (v41 & 1) != 0 )
  {
    v97 = v25;
    sub_6F40C0((__int64)a1, a2, a3, v23, a5, v102);
    v56 = sub_6F6F40((const __m128i *)a1, 0, v52, v53, v54, v55);
    v57 = sub_6F5430(0, v56, (__int64)v12, 0, 0, 0, 0, 0, 0, 0, (__int64)a1 + 68);
    v25 = v97;
    v103 = v57;
    a5 = v57;
    goto LABEL_25;
  }
  if ( v95 )
  {
    v42 = (__m128i *)a1;
    v86 = v25;
    v43 = (_QWORD *)((char *)a1 + 68);
    sub_8441D0(v42, v95, (v41 & 4) != 0, a4, &v104, &v102);
    if ( !v102 )
    {
      v72 = sub_6F5430(v95, v104, (__int64)v12, 0, 0, 0, 0, 0, 1u, 0, (__int64)v43);
      v25 = v86;
      v103 = v72;
      a5 = v72;
      if ( v93 )
      {
        sub_6EB360(v72, v13, v13, v43);
        a5 = v103;
        v25 = v86;
      }
LABEL_25:
      if ( !v25 )
        goto LABEL_27;
      goto LABEL_26;
    }
    if ( *(_BYTE *)(v104 + 24) != 2 )
    {
      if ( v93 )
        v44 = sub_6EB460(3u, v13, v43);
      else
        v44 = sub_6EAFA0(3u);
      v25 = v86;
      a5 = v44;
      v103 = v44;
      *(_QWORD *)(v44 + 56) = v104;
      goto LABEL_87;
    }
    v100 = v86;
    if ( v93 )
      v70 = sub_6EB460(2u, v13, v43);
    else
      v70 = sub_6EAFA0(2u);
    v103 = v70;
    v68 = *(const __m128i **)(v104 + 56);
LABEL_142:
    v71 = sub_740630(v68);
    sub_72F900(v103, v71);
    a5 = v103;
    v25 = v100;
    goto LABEL_25;
  }
  v103 = 0;
  if ( v25 )
  {
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) = (v18 << 6)
                                                              | *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13)
                                                              & 0xBF;
    if ( !a9 )
      goto LABEL_47;
    goto LABEL_29;
  }
  v31 = 0;
  if ( a9 )
    goto LABEL_29;
LABEL_48:
  if ( a7 )
    *a7 = v10;
  *a8 = v31;
  return a8;
}
