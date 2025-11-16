// Function: sub_8C4EC0
// Address: 0x8c4ec0
//
__int64 __fastcall sub_8C4EC0(__int64 a1, _DWORD *a2)
{
  char v3; // al
  __int64 v4; // r14
  __int64 **v5; // r13
  __int64 *v6; // rbx
  _QWORD *i; // rbx
  __int64 v8; // r14
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r15
  int v14; // ebx
  __int64 v15; // r15
  __int64 v16; // r14
  char v17; // al
  _QWORD *v18; // rax
  __int64 *v19; // rax
  __int64 *v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // r15
  __int64 v23; // r14
  _QWORD *v24; // rax
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 v27; // rsi
  __int64 *v28; // rax
  __int64 *v29; // rcx
  __int64 v30; // rdx
  __int64 **v31; // rdx
  __int64 v32; // r14
  __int64 *v33; // r15
  _QWORD *v34; // rax
  __int64 v35; // rax
  __int64 *v36; // r14
  __int64 *v37; // r15
  __int64 v38; // rdi
  __int64 v39; // rax
  _QWORD *v40; // rdx
  __int64 *v41; // r13
  char v42; // al
  __int64 **v43; // rax
  __int64 v44; // r14
  _QWORD *v45; // rax
  int v46; // ebx
  __int64 v48; // rdi
  __int64 *v49; // rax
  __int64 v50; // r13
  __int64 v51; // rbx
  char v52; // al
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rbx
  char v56; // al
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdi
  _QWORD *v60; // rax
  __int64 v61; // rax
  __int64 *v62; // rax
  __int64 v63; // rsi
  __int64 *v64; // rax
  __int64 v65; // rsi
  unsigned int v66; // eax
  _QWORD *v67; // rax
  char v68; // al
  __int64 v69; // rbx
  _QWORD *v70; // rax
  _QWORD *v71; // rax
  char v72; // al
  __int64 v73; // rbx
  _QWORD *v74; // rax
  __int64 v75; // rax
  char v76; // al
  char v77; // al
  __int64 v78; // rbx
  char v79; // al
  __int64 *v80; // rdi
  __int64 v81; // r12
  _QWORD *v82; // rax
  __int64 *v83; // rbx
  __m128i *v84; // rax
  __int64 v85; // r13
  _QWORD *v86; // rax
  __int64 v87; // r13
  _QWORD *v88; // rax
  char v89; // al
  __int64 *v90; // r12
  char v91; // al
  __int64 v92; // rbx
  _QWORD *v93; // rax
  __int64 *v94; // rdi
  __int64 v95; // r13
  _QWORD *v96; // rax
  __int64 v97; // r13
  _QWORD *v98; // rax
  __int64 v99; // [rsp+8h] [rbp-48h]
  int v100; // [rsp+10h] [rbp-40h]
  unsigned int v101; // [rsp+14h] [rbp-3Ch]

  v3 = *(_BYTE *)(a1 + 28);
  if ( !v3 )
  {
    v51 = qword_4D03FD0[1];
    v52 = *(_BYTE *)(a1 - 8);
    *(_QWORD *)(a1 - 24) = v51;
    if ( (v52 & 8) == 0 )
    {
      *(_BYTE *)(a1 - 8) = v52 | 8;
      sub_8C3650((__int64 *)a1, 0x17u, 1);
      v85 = *(_QWORD *)(a1 - 24);
      v86 = sub_7247C0(qword_4B6D500[23]);
      *(_QWORD *)(a1 - 24) = v86;
      *(v86 - 3) = v85;
      *(_BYTE *)(a1 - 8) |= 4u;
    }
    v53 = *(_QWORD *)(a1 + 88);
    if ( v53 )
    {
      v54 = *(_QWORD *)(v51 + 88);
      if ( v54 )
      {
        *(_QWORD *)(v53 - 24) = v54;
        v55 = *(_QWORD *)(a1 + 88);
        v56 = *(_BYTE *)(v55 - 8);
        if ( (v56 & 8) == 0 )
        {
          v94 = *(__int64 **)(a1 + 88);
          *(_BYTE *)(v55 - 8) = v56 | 8;
          sub_8C3650(v94, 0x16u, 1);
          v95 = *(_QWORD *)(v55 - 24);
          v96 = sub_7247C0(qword_4B6D500[22]);
          *(_QWORD *)(v55 - 24) = v96;
          *(v96 - 3) = v95;
          *(_BYTE *)(v55 - 8) |= 4u;
        }
      }
    }
    v57 = sub_85EB10(a1);
    v8 = *(_QWORD *)(a1 + 104);
    v100 = 1;
    v101 = 1;
    v5 = (__int64 **)v57;
    if ( v8 )
      goto LABEL_117;
    goto LABEL_14;
  }
  v4 = *(_QWORD *)(a1 + 32);
  if ( v3 == 3 )
  {
    v9 = *(__int64 **)(v4 + 32);
    if ( v9 && ((v10 = *v9, *v9 != v4) || v9[1]) )
    {
      v11 = *(_QWORD *)(v10 + 128);
      if ( (*(_BYTE *)(v11 - 8) & 3) == 3 )
      {
        sub_8C3650(*(__int64 **)(v10 + 128), 0x17u, 0);
        v11 = *(_QWORD *)(v11 - 24);
        if ( (*(_BYTE *)(v11 - 8) & 2) != 0 )
          v11 = *(_QWORD *)(v11 - 24);
      }
      *(_QWORD *)(a1 - 24) = v11;
      v100 = 1;
      v101 = 0;
    }
    else
    {
      v100 = 0;
      v101 = 1;
    }
    v12 = sub_85EB10(a1);
    v8 = *(_QWORD *)(a1 + 104);
    v5 = (__int64 **)v12;
    if ( v8 )
      goto LABEL_117;
LABEL_14:
    v13 = 0;
    v14 = 0;
    goto LABEL_15;
  }
  v5 = *(__int64 ***)(v4 + 32);
  if ( !v5 )
    goto LABEL_192;
  v6 = *v5;
  if ( *v5 == (__int64 *)v4 )
  {
    v5 = (__int64 **)v5[1];
    if ( v5 )
    {
      v89 = *((_BYTE *)v6 - 8);
      if ( (v89 & 8) != 0 )
      {
        v8 = *(_QWORD *)(a1 + 104);
        if ( !v8 )
        {
          v15 = *(_QWORD *)(a1 + 112);
          if ( !v15 )
          {
            v101 = 1;
            v14 = 0;
            v100 = 0;
            goto LABEL_178;
          }
          v101 = 1;
          v5 = 0;
          v14 = 0;
          v100 = 0;
          goto LABEL_17;
        }
        goto LABEL_199;
      }
      *((_BYTE *)v6 - 8) = v89 | 8;
      sub_8C3650(v6, 6u, 1);
      v97 = *(v6 - 3);
      v98 = sub_7247C0(qword_4B6D500[6]);
      *(v6 - 3) = (__int64)v98;
      *(v98 - 3) = v97;
      *((_BYTE *)v6 - 8) |= 4u;
      v8 = *(_QWORD *)(a1 + 104);
      if ( v8 )
      {
LABEL_199:
        v5 = 0;
        goto LABEL_193;
      }
LABEL_208:
      v100 = 0;
      v14 = 0;
      v101 = 1;
LABEL_177:
      v15 = *(_QWORD *)(a1 + 112);
      v5 = 0;
      if ( !v15 )
        goto LABEL_178;
      goto LABEL_17;
    }
LABEL_192:
    v8 = *(_QWORD *)(a1 + 104);
    if ( v8 )
    {
LABEL_193:
      v100 = 0;
      v101 = 1;
      goto LABEL_117;
    }
    goto LABEL_208;
  }
  if ( (unsigned int)sub_8D2490(*v5) )
  {
    v83 = *(__int64 **)(v6[21] + 152);
    if ( (*(_BYTE *)(v83 - 1) & 3) == 3 )
    {
      sub_8C3650(v83, 0x17u, 0);
      v83 = (__int64 *)*(v83 - 3);
      if ( (*(_BYTE *)(v83 - 1) & 2) != 0 )
        v83 = (__int64 *)*(v83 - 3);
    }
    *(_QWORD *)(a1 - 24) = v83;
  }
  for ( i = *(_QWORD **)(v4 + 160); i; i = (_QWORD *)i[14] )
  {
    v49 = (__int64 *)i[4];
    v50 = (__int64)i;
    if ( v49 )
      v50 = *v49;
    sub_8C2B90((__int64)i, v50);
    if ( !*(_QWORD *)(v50 + 152) )
    {
      v48 = i[19];
      if ( v48 )
      {
        if ( dword_4D03FE8[0] )
        {
          *(_QWORD *)(v50 + 152) = sub_740B80(v48, 0);
        }
        else
        {
          dword_4D03FE8[0] = 1;
          sub_727950();
          v84 = sub_740B80(i[19], 0);
          dword_4D03FE8[0] = 0;
          *(_QWORD *)(v50 + 152) = v84;
          sub_727950();
        }
      }
    }
  }
  *(_QWORD *)(v4 + 160) = 0;
  v8 = *(_QWORD *)(a1 + 104);
  if ( !v8 )
  {
    v100 = 1;
    v14 = 0;
    v101 = 0;
    goto LABEL_177;
  }
  v100 = 1;
  v5 = 0;
  v101 = 0;
LABEL_117:
  v13 = 0;
  v14 = 0;
  if ( (unsigned __int8)(*(_BYTE *)(v8 + 140) - 9) <= 2u )
    goto LABEL_118;
LABEL_124:
  v60 = *(_QWORD **)(v8 + 32);
  if ( !v60 )
    goto LABEL_121;
  if ( *v60 != v8 )
  {
LABEL_126:
    v61 = *(_QWORD *)(v8 + 112);
    if ( v13 )
      *(_QWORD *)(v13 + 112) = v61;
    else
      *(_QWORD *)(a1 + 104) = v61;
    v62 = *(__int64 **)(v8 + 32);
    v63 = v8;
    if ( v62 )
      v63 = *v62;
    sub_8C3930(v8, v63);
    goto LABEL_122;
  }
  if ( v60[1] )
  {
    v72 = *(_BYTE *)(v8 - 8);
    if ( (v72 & 8) == 0 )
    {
      *(_BYTE *)(v8 - 8) = v72 | 8;
      sub_8C3650((__int64 *)v8, 6u, 1);
      v73 = *(_QWORD *)(v8 - 24);
      v74 = sub_7247C0(qword_4B6D500[6]);
      *(_QWORD *)(v8 - 24) = v74;
      *(v74 - 3) = v73;
      *(_BYTE *)(v8 - 8) |= 4u;
    }
  }
  while ( 1 )
  {
LABEL_121:
    v13 = v8;
    v14 = 1;
    if ( dword_4F077C4 == 2 && (unsigned __int8)(*(_BYTE *)(v8 + 140) - 9) <= 2u )
    {
      v75 = *(_QWORD *)(v8 + 168);
      if ( v75 )
        *(_QWORD *)(v75 + 128) = 0;
    }
LABEL_122:
    v8 = *(_QWORD *)(v8 + 112);
    if ( !v8 )
      break;
    if ( (unsigned __int8)(*(_BYTE *)(v8 + 140) - 9) > 2u )
      goto LABEL_124;
LABEL_118:
    v58 = *(_QWORD *)(v8 + 168);
    if ( !v58 )
      goto LABEL_124;
    v59 = *(_QWORD *)(v58 + 152);
    if ( !v59 )
      goto LABEL_124;
    if ( !(unsigned int)sub_8C4EC0(v59, a2) )
      goto LABEL_126;
  }
LABEL_15:
  if ( !v5 )
    goto LABEL_177;
  v5[4] = (__int64 *)v13;
  v15 = *(_QWORD *)(a1 + 112);
  if ( v15 )
  {
LABEL_17:
    v16 = 0;
    while ( 1 )
    {
      if ( unk_4D03FD8 )
      {
        if ( (*(_BYTE *)(v15 + 170) & 0x90) != 0x10 )
        {
          sub_735A70(v15);
          if ( sub_7E16F0() && *(_BYTE *)(v15 + 136) == 2 )
          {
            sub_7E99A0(v15);
            *(_BYTE *)(v15 + 136) = 1;
          }
        }
      }
      v18 = *(_QWORD **)(v15 + 32);
      if ( !v18 )
        goto LABEL_29;
      if ( *v18 == v15 )
      {
        if ( v18[1] )
          goto LABEL_20;
LABEL_29:
        v16 = v15;
        v15 = *(_QWORD *)(v15 + 112);
        v14 = 1;
        if ( !v15 )
        {
LABEL_30:
          if ( v5 )
            goto LABEL_31;
LABEL_178:
          v19 = *(__int64 **)(a1 + 192);
          if ( v19 )
          {
            v5 = 0;
            goto LABEL_32;
          }
LABEL_180:
          v22 = *(_QWORD *)(a1 + 144);
          if ( !v22 )
            goto LABEL_184;
          v5 = 0;
          goto LABEL_40;
        }
      }
      else
      {
        if ( !*(_QWORD *)(v15 + 104) )
        {
          if ( v16 )
            *(_QWORD *)(v16 + 112) = *(_QWORD *)(v15 + 112);
          else
            *(_QWORD *)(a1 + 112) = *(_QWORD *)(v15 + 112);
          v64 = *(__int64 **)(v15 + 32);
          v65 = v15;
          if ( v64 )
            v65 = *v64;
          *(_BYTE *)(v65 + 169) = (*(_BYTE *)(v65 + 169) | *(_BYTE *)(v15 + 169)) & 0x40 | *(_BYTE *)(v65 + 169) & 0xBF;
          sub_8C2B90(v15, v65);
          v66 = *(_DWORD *)(v15 + 152);
          if ( v66 > *(_DWORD *)(v65 + 152) )
            *(_DWORD *)(v65 + 152) = v66;
          if ( !*(_BYTE *)(v15 + 136) )
            sub_735A70(v15);
          sub_734EF0(v15);
          goto LABEL_22;
        }
LABEL_20:
        v17 = *(_BYTE *)(v15 - 8);
        v16 = v15;
        v14 = 1;
        if ( (v17 & 8) == 0 )
        {
          *(_BYTE *)(v15 - 8) = v17 | 8;
          sub_8C3650((__int64 *)v15, 7u, 1);
          v99 = *(_QWORD *)(v15 - 24);
          v67 = sub_7247C0(qword_4B6D500[7]);
          *(_QWORD *)(v15 - 24) = v67;
          *(v67 - 3) = v99;
          *(_BYTE *)(v15 - 8) |= 4u;
        }
LABEL_22:
        v15 = *(_QWORD *)(v15 + 112);
        if ( !v15 )
          goto LABEL_30;
      }
    }
  }
  v16 = 0;
LABEL_31:
  v5[5] = (__int64 *)v16;
  v19 = *(__int64 **)(a1 + 192);
  if ( v19 )
  {
LABEL_32:
    v20 = 0;
    do
    {
      while ( *(_BYTE *)(v19[1] + 177) )
      {
        v20 = v19;
        v19 = (__int64 *)*v19;
        if ( !v19 )
          goto LABEL_38;
      }
      v21 = *v19;
      if ( v20 )
        *v20 = v21;
      else
        *(_QWORD *)(a1 + 192) = v21;
      v19 = (__int64 *)*v19;
    }
    while ( v19 );
LABEL_38:
    if ( !v5 )
      goto LABEL_180;
  }
  else
  {
    v20 = 0;
  }
  v5[8] = v20;
  v22 = *(_QWORD *)(a1 + 144);
  if ( v22 )
  {
LABEL_40:
    v23 = 0;
    while ( 1 )
    {
      if ( (unsigned int)sub_89A120(v22) && sub_7E16F0() && *(_BYTE *)(v22 + 172) == 2 )
      {
        sub_7E99A0(v22);
        v24 = *(_QWORD **)(v22 + 32);
        *(_BYTE *)(v22 + 172) = *(_DWORD *)(v22 + 160) == 0;
        if ( !v24 )
          goto LABEL_54;
      }
      else
      {
        v24 = *(_QWORD **)(v22 + 32);
        if ( !v24 )
          goto LABEL_54;
      }
      if ( *v24 != v22 )
      {
        v25 = *(_QWORD *)(v22 + 112);
        if ( v23 )
          *(_QWORD *)(v23 + 112) = v25;
        else
          *(_QWORD *)(a1 + 144) = v25;
        v26 = *(__int64 **)(v22 + 32);
        v27 = v22;
        if ( v26 )
          v27 = *v26;
        sub_8C2DF0(v22, v27);
        if ( *(_DWORD *)(v22 + 160) )
        {
          v71 = (_QWORD *)sub_72B840(v22);
          sub_734690(v71);
          if ( (*(_BYTE *)(v22 + 195) & 2) == 0 )
            sub_8C3FF0(v22);
          *a2 = 1;
        }
        sub_734AA0(v22);
        goto LABEL_49;
      }
      if ( v24[1] )
      {
        v68 = *(_BYTE *)(v22 - 8);
        if ( (v68 & 8) == 0 )
        {
          *(_BYTE *)(v22 - 8) = v68 | 8;
          sub_8C3650((__int64 *)v22, 0xBu, 1);
          v69 = *(_QWORD *)(v22 - 24);
          v70 = sub_7247C0(qword_4B6D500[11]);
          *(_QWORD *)(v22 - 24) = v70;
          *(v70 - 3) = v69;
          *(_BYTE *)(v22 - 8) |= 4u;
        }
      }
LABEL_54:
      if ( (*(_BYTE *)(v22 + 194) & 0x40) != 0 )
      {
        v23 = v22;
        v14 = 1;
LABEL_49:
        v22 = *(_QWORD *)(v22 + 112);
        if ( !v22 )
          goto LABEL_56;
      }
      else
      {
        *(_QWORD *)(v22 + 232) = 0;
        v23 = v22;
        v22 = *(_QWORD *)(v22 + 112);
        v14 = 1;
        if ( !v22 )
        {
LABEL_56:
          if ( v5 )
            goto LABEL_57;
LABEL_184:
          v28 = *(__int64 **)(a1 + 272);
          if ( !v28 )
            goto LABEL_182;
          v5 = 0;
LABEL_58:
          v29 = 0;
          while ( 1 )
          {
            v31 = (__int64 **)v28[4];
            if ( v31 )
            {
              if ( *v31 != v28 )
              {
                v30 = v28[14];
                if ( v29 )
                  v29[14] = v30;
                else
                  *(_QWORD *)(a1 + 272) = v30;
LABEL_62:
                v28 = (__int64 *)v28[14];
                if ( !v28 )
                  goto LABEL_65;
                continue;
              }
              if ( v31[1] )
              {
                v31[1] = 0;
                v29 = v28;
                v14 = 1;
                goto LABEL_62;
              }
            }
            v29 = v28;
            v28 = (__int64 *)v28[14];
            v14 = 1;
            if ( !v28 )
            {
LABEL_65:
              if ( v5 )
                goto LABEL_66;
LABEL_182:
              v32 = *(_QWORD *)(a1 + 168);
              if ( !v32 )
                goto LABEL_174;
              v5 = 0;
LABEL_67:
              v33 = 0;
              while ( 2 )
              {
                if ( (*(_BYTE *)(v32 + 124) & 1) != 0 )
                {
                  v34 = *(_QWORD **)(v32 + 32);
                  if ( v34 )
                  {
                    if ( *v34 == v32 && !v34[1] )
                    {
                      v33 = (__int64 *)v32;
                      v14 = 1;
                      goto LABEL_73;
                    }
LABEL_71:
                    v35 = *(_QWORD *)(v32 + 112);
                    if ( v33 )
                      v33[14] = v35;
                    else
                      *(_QWORD *)(a1 + 168) = v35;
LABEL_73:
                    v32 = *(_QWORD *)(v32 + 112);
                    if ( !v32 )
                    {
                      if ( v5 )
                        goto LABEL_75;
LABEL_174:
                      v36 = *(__int64 **)(a1 + 232);
                      if ( !v36 )
                        goto LABEL_87;
                      v5 = 0;
LABEL_76:
                      v37 = 0;
                      while ( 2 )
                      {
                        while ( 1 )
                        {
                          v38 = v36[3];
                          if ( v38 )
                          {
                            v39 = sub_72A270(v38, *((_BYTE *)v36 + 16));
                            if ( v39 )
                            {
                              v40 = *(_QWORD **)(v39 + 32);
                              if ( !v40 || *v40 == v39 )
                                break;
                            }
                          }
                          if ( v37 )
                          {
                            *v37 = *v36;
                            goto LABEL_78;
                          }
                          *(_QWORD *)(a1 + 232) = *v36;
                          v36 = (__int64 *)*v36;
                          if ( !v36 )
                          {
LABEL_85:
                            if ( v5 )
                              goto LABEL_86;
                            goto LABEL_87;
                          }
                        }
                        v37 = v36;
                        v14 = 1;
LABEL_78:
                        v36 = (__int64 *)*v36;
                        if ( !v36 )
                          goto LABEL_85;
                        continue;
                      }
                    }
                    continue;
                  }
                }
                else if ( !(unsigned int)sub_8C4EC0(*(_QWORD *)(v32 + 128), a2) )
                {
                  goto LABEL_71;
                }
                break;
              }
              v33 = (__int64 *)v32;
              v14 = 1;
              goto LABEL_73;
            }
          }
        }
      }
    }
  }
  v23 = 0;
LABEL_57:
  v5[6] = (__int64 *)v23;
  v28 = *(__int64 **)(a1 + 272);
  if ( v28 )
    goto LABEL_58;
  v29 = 0;
LABEL_66:
  v5[13] = v29;
  v32 = *(_QWORD *)(a1 + 168);
  if ( v32 )
    goto LABEL_67;
  v33 = 0;
LABEL_75:
  v5[9] = v33;
  v36 = *(__int64 **)(a1 + 232);
  if ( v36 )
    goto LABEL_76;
  v37 = 0;
LABEL_86:
  v5[12] = v37;
LABEL_87:
  if ( !*(_BYTE *)(a1 + 28) )
  {
    if ( *a2 && dword_4F06C5C )
      sub_735030();
    v41 = (__int64 *)qword_4F07300;
    if ( qword_4F07300 )
    {
      while ( 1 )
      {
        v43 = (__int64 **)v41[4];
        if ( !v43 )
          goto LABEL_94;
        if ( *v43 == v41 )
        {
          if ( v43[1] )
          {
            v42 = *((_BYTE *)v41 - 8);
            if ( (v42 & 8) == 0 )
              goto LABEL_99;
          }
LABEL_94:
          v41 = (__int64 *)v41[14];
          if ( !v41 )
            break;
        }
        else
        {
          v42 = *((_BYTE *)v41 - 8);
          if ( (v42 & 8) != 0 )
            goto LABEL_94;
LABEL_99:
          *((_BYTE *)v41 - 8) = v42 | 8;
          sub_8C3650(v41, 6u, 1);
          v44 = *(v41 - 3);
          v45 = sub_7247C0(qword_4B6D500[6]);
          *(v41 - 3) = (__int64)v45;
          *(v45 - 3) = v44;
          *((_BYTE *)v41 - 8) |= 4u;
          v41 = (__int64 *)v41[14];
          if ( !v41 )
            break;
        }
      }
    }
  }
  v46 = v100 & v14;
  if ( v46 )
  {
    v76 = *(_BYTE *)(a1 - 8);
    if ( (v76 & 8) == 0 )
    {
      *(_BYTE *)(a1 - 8) = v76 | 8;
      sub_8C3650((__int64 *)a1, 0x17u, 1);
      v87 = *(_QWORD *)(a1 - 24);
      v88 = sub_7247C0(qword_4B6D500[23]);
      *(_QWORD *)(a1 - 24) = v88;
      *(v88 - 3) = v87;
      *(_BYTE *)(a1 - 8) |= 4u;
    }
    v77 = *(_BYTE *)(a1 + 28);
    if ( v77 == 3 )
    {
      v90 = *(__int64 **)(a1 + 32);
      v101 = v46;
      v91 = *((_BYTE *)v90 - 8);
      if ( (v91 & 8) == 0 )
      {
        *((_BYTE *)v90 - 8) = v91 | 8;
        sub_8C3650(v90, 0x1Cu, 1);
        v92 = *(v90 - 3);
        v93 = sub_7247C0(qword_4B6D500[28]);
        *(v90 - 3) = (__int64)v93;
        *(v93 - 3) = v92;
        *((_BYTE *)v90 - 8) |= 4u;
      }
    }
    else
    {
      v101 = v46;
      if ( v77 == 6 )
      {
        v78 = *(_QWORD *)(a1 + 32);
        v79 = *(_BYTE *)(v78 - 8);
        if ( (v79 & 8) == 0 )
        {
          v80 = *(__int64 **)(a1 + 32);
          *(_BYTE *)(v78 - 8) = v79 | 8;
          sub_8C3650(v80, 6u, 1);
          v81 = *(_QWORD *)(v78 - 24);
          v82 = sub_7247C0(qword_4B6D500[6]);
          *(_QWORD *)(v78 - 24) = v82;
          *(v82 - 3) = v81;
          *(_BYTE *)(v78 - 8) |= 4u;
        }
      }
    }
  }
  return v101;
}
