// Function: sub_8E3AD0
// Address: 0x8e3ad0
//
__int64 __fastcall sub_8E3AD0(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 *v4; // r9
  __int64 v5; // r15
  int i; // eax
  unsigned int v7; // r12d
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r13
  char v12; // al
  int v13; // r14d
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // r15
  unsigned __int64 *v20; // rbx
  unsigned __int64 *v21; // r12
  int v22; // r13d
  unsigned __int64 v23; // r14
  __int64 v24; // rdi
  __int64 v25; // r13
  int v26; // r14d
  char v27; // al
  __int64 v28; // rdx
  __int64 v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // r12
  __int64 *v32; // rax
  __int64 v33; // r12
  __int64 *v34; // rax
  unsigned __int64 v35; // r15
  unsigned __int64 *v36; // rbx
  unsigned __int64 *v37; // r12
  int v38; // r13d
  unsigned __int64 v39; // r14
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 j; // rbx
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rbx
  __int64 *v46; // rax
  __int64 v47; // rbx
  __int64 v48; // rdi
  char k; // al
  unsigned int v50; // [rsp+Ch] [rbp-114h]
  __int64 v51; // [rsp+10h] [rbp-110h]
  unsigned int v52; // [rsp+10h] [rbp-110h]
  __int64 v53; // [rsp+28h] [rbp-F8h]
  __int64 v54; // [rsp+28h] [rbp-F8h]
  unsigned __int64 *v55; // [rsp+30h] [rbp-F0h]
  unsigned __int64 *v56; // [rsp+30h] [rbp-F0h]
  unsigned __int64 *v57; // [rsp+38h] [rbp-E8h]
  unsigned __int64 *v58; // [rsp+38h] [rbp-E8h]
  __int64 v59; // [rsp+38h] [rbp-E8h]
  int v60; // [rsp+48h] [rbp-D8h] BYREF
  int v61; // [rsp+4Ch] [rbp-D4h] BYREF
  _BYTE v62[4]; // [rsp+50h] [rbp-D0h] BYREF
  int v63; // [rsp+54h] [rbp-CCh]
  _BYTE v64[40]; // [rsp+58h] [rbp-C8h] BYREF
  unsigned __int64 *v65; // [rsp+80h] [rbp-A0h]
  __int64 v66; // [rsp+88h] [rbp-98h]
  __int64 v67; // [rsp+90h] [rbp-90h]
  _BYTE v68[4]; // [rsp+A0h] [rbp-80h] BYREF
  int v69; // [rsp+A4h] [rbp-7Ch]
  _BYTE v70[40]; // [rsp+A8h] [rbp-78h] BYREF
  unsigned __int64 *v71; // [rsp+D0h] [rbp-50h]
  __int64 v72; // [rsp+D8h] [rbp-48h]
  __int64 v73; // [rsp+E0h] [rbp-40h]

  if ( (*(_BYTE *)(a1 + 140) & 0xFB) == 8 )
  {
    a2 = dword_4F077C4 != 2;
    if ( (sub_8D4C10(a1, a2) & 2) != 0 )
    {
      if ( (_DWORD)qword_4F077B4 )
      {
        if ( (unsigned __int64)(qword_4F077A0 - 30400LL) <= 0x2580 )
          return 0;
      }
      else if ( dword_4F077BC && qword_4F077A8 <= 0x1869Fu && !sub_8D3A70(a1) )
      {
        return 0;
      }
    }
  }
  v5 = sub_8D4130(a1);
  for ( i = *(unsigned __int8 *)(v5 + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v5 + 140) )
    v5 = *(_QWORD *)(v5 + 160);
  v7 = 1;
  if ( (unsigned __int8)(i - 2) > 3u )
  {
    if ( (_BYTE)i == 6 )
    {
      if ( (*(_BYTE *)(v5 + 168) & 1) == 0 )
        return v7;
    }
    else
    {
      if ( (unsigned __int8)(i - 19) <= 1u || (_BYTE)i == 13 )
        return 1;
      v9 = (unsigned int)(i - 9);
      if ( (unsigned __int8)(i - 9) <= 2u )
      {
        if ( !*(_QWORD *)v5 )
          return 0;
        v10 = *(_QWORD *)(*(_QWORD *)v5 + 96LL);
        v53 = v10;
        if ( *(_QWORD *)(v10 + 24) )
        {
          if ( (*(_BYTE *)(v10 + 177) & 2) == 0 )
            return 0;
        }
        v72 = 0;
        v65 = (unsigned __int64 *)v64;
        v71 = (unsigned __int64 *)v70;
        v66 = 0;
        v67 = 0;
        v73 = 0;
        v11 = *(_QWORD *)(v10 + 8);
        v63 = 1;
        v69 = 1;
        if ( v11 )
        {
          v12 = *(_BYTE *)(v11 + 80);
          v13 = 0;
          if ( v12 != 17 )
            goto LABEL_24;
          v11 = *(_QWORD *)(v11 + 88);
          if ( dword_4F077C4 != 2 )
          {
            if ( v11 )
            {
LABEL_122:
              v12 = *(_BYTE *)(v11 + 80);
              v13 = 1;
LABEL_24:
              v7 = 0;
              while ( 1 )
              {
                if ( v12 != 20
                  && !((*(_BYTE *)(v11 + 104) & 1) != 0
                     ? sub_8796F0(v11)
                     : (*(_BYTE *)(*(_QWORD *)(v11 + 88) + 208LL) & 4) != 0) )
                {
                  v15 = *(_QWORD *)(v11 + 88);
                  if ( (*(_BYTE *)(v15 + 194) & 4) != 0 )
                  {
                    v33 = v67;
                    if ( v67 == v66 )
                      sub_8E3990((__int64)v62, a2, v9, v2, v3, v4);
                    v34 = (__int64 *)&v65[v33];
                    if ( v34 )
                      *v34 = v11;
                    v67 = v33 + 1;
                    v7 = 1;
                  }
                  else if ( (*(_BYTE *)(v15 + 206) & 0x10) == 0 )
                  {
                    v16 = **(_QWORD **)(*(_QWORD *)(v15 + 152) + 168LL);
                    if ( v16 )
                    {
                      if ( !*(_QWORD *)v16 || (*(_BYTE *)(v16 + 32) & 4) != 0 )
                      {
                        a2 = v5;
                        if ( (unsigned int)sub_72F500(v15, v5, 0, 1, 1) )
                        {
                          v41 = v73;
                          if ( v73 == v72 )
                          {
                            v59 = v73;
                            sub_8E3990((__int64)v68, v5, v9, v2, v3, v4);
                            v41 = v59;
                          }
                          v9 = (__int64)&v71[v41];
                          if ( v9 )
                            *(_QWORD *)v9 = v11;
                          v73 = v41 + 1;
                        }
                      }
                    }
                  }
                }
                if ( !v13 )
                  break;
                v11 = *(_QWORD *)(v11 + 8);
                if ( !v11 )
                  break;
                v12 = *(_BYTE *)(v11 + 80);
              }
LABEL_36:
              v17 = v67;
              v18 = v73;
              if ( v67 )
              {
                v55 = &v71[v73];
                if ( v55 != v71 )
                {
                  v57 = v71;
                  v51 = v5;
                  v50 = v7;
                  do
                  {
                    v19 = *v57;
                    v2 = (__int64)v65;
                    v20 = &v65[v17];
                    if ( v20 == v65 )
                      goto LABEL_80;
                    v21 = v65;
                    v22 = 0;
                    do
                    {
                      v23 = *v21;
                      a2 = v19;
                      if ( (int)sub_6F3270(*v21, v19, 0) > 0 )
                      {
                        v24 = *(_QWORD *)(*(_QWORD *)(v19 + 88) + 152LL);
                        a2 = *(_QWORD *)(*(_QWORD *)(v23 + 88) + 152LL);
                        if ( v24 == a2 || (unsigned int)sub_8D97D0(v24, a2, 0, v2, v3) )
                          v22 = 1;
                      }
                      ++v21;
                    }
                    while ( v20 != v21 );
                    if ( !v22 )
                      goto LABEL_80;
                    ++v57;
                    v17 = v67;
                  }
                  while ( v55 != v57 );
                  v5 = v51;
                  v7 = v50;
                  v18 = v73;
                }
                v2 = v53;
                v25 = *(_QWORD *)(v53 + 32);
                if ( !v25 )
                {
                  v26 = 0;
                  if ( v17 <= 0 )
                  {
                    if ( v18 <= 0 )
                      goto LABEL_66;
                    v73 = 0;
                    goto LABEL_151;
                  }
LABEL_51:
                  a2 = 0;
                  v2 = 1 - v17;
                  v67 = 0;
LABEL_52:
                  if ( v18 > 0 )
                    v73 = 0;
                  if ( v25 )
                  {
                    while ( 1 )
                    {
                      v27 = *(_BYTE *)(v25 + 80);
                      v28 = v25;
                      if ( v27 == 16 )
                      {
                        v28 = **(_QWORD **)(v25 + 88);
                        v27 = *(_BYTE *)(v28 + 80);
                      }
                      if ( v27 == 24 )
                      {
                        v28 = *(_QWORD *)(v28 + 88);
                        v27 = *(_BYTE *)(v28 + 80);
                      }
                      if ( v27 != 20 )
                      {
                        v29 = *(_QWORD *)(v28 + 88);
                        if ( (*(_BYTE *)(v29 + 194) & 4) != 0 )
                        {
                          v31 = v67;
                          if ( v67 == v66 )
                            sub_8E3990((__int64)v62, a2, v28, v2, v3, v4);
                          v32 = (__int64 *)&v65[v31];
                          if ( v32 )
                            *v32 = v25;
                          v67 = v31 + 1;
                          v7 = 1;
                        }
                        else if ( (*(_BYTE *)(v29 + 206) & 0x10) == 0 )
                        {
                          a2 = (__int64)&v61;
                          if ( (unsigned int)sub_72F790(v29, &v61, &v60) )
                          {
                            v45 = v73;
                            if ( v73 == v72 )
                              sub_8E3990((__int64)v68, (__int64)&v61, v30, v2, v3, v4);
                            v46 = (__int64 *)&v71[v45];
                            if ( v46 )
                              *v46 = v25;
                            v18 = v45 + 1;
                            v17 = v67;
                            v73 = v45 + 1;
LABEL_66:
                            if ( !v17 )
                            {
                              if ( !v18 )
                                goto LABEL_68;
LABEL_80:
                              if ( v71 != (unsigned __int64 *)v70 )
                                sub_823A00((__int64)v71, 8 * v72, v18, v2, v3, v4);
                              if ( v65 != (unsigned __int64 *)v64 )
                                sub_823A00((__int64)v65, 8 * v66, v18, v2, v3, v4);
                              return 0;
                            }
                            v2 = (__int64)&v71[v18];
                            v56 = (unsigned __int64 *)v2;
                            if ( (unsigned __int64 *)v2 != v71 )
                            {
                              v58 = v71;
                              v54 = v5;
                              v52 = v7;
                              while ( 1 )
                              {
                                v35 = *v58;
                                v2 = (__int64)v65;
                                v36 = &v65[v17];
                                if ( v36 == v65 )
                                  goto LABEL_80;
                                v37 = v65;
                                v38 = 0;
                                do
                                {
                                  v39 = *v37;
                                  if ( (int)sub_6F3270(*v37, v35, 0) > 0 )
                                  {
                                    v4 = *(__int64 **)(*(_QWORD *)(v35 + 88) + 152LL);
                                    v40 = *(_QWORD *)(*(_QWORD *)(v39 + 88) + 152LL);
                                    if ( v4 == (__int64 *)v40
                                      || (unsigned int)sub_8D97D0(
                                                         *(_QWORD *)(*(_QWORD *)(v35 + 88) + 152LL),
                                                         v40,
                                                         0,
                                                         v2,
                                                         v3) )
                                    {
                                      v38 = 1;
                                    }
                                  }
                                  ++v37;
                                }
                                while ( v36 != v37 );
                                if ( !v38 )
                                  goto LABEL_80;
                                if ( v56 == ++v58 )
                                {
                                  v5 = v54;
                                  v7 = v52;
                                  break;
                                }
                                v17 = v67;
                              }
                            }
LABEL_68:
                            if ( v7 )
                            {
                              if ( (_DWORD)qword_4F077B4 )
                              {
                                if ( (*(_BYTE *)(v5 + 176) & 2) != 0 )
                                {
                                  for ( j = *(_QWORD *)(v5 + 160); j; j = *(_QWORD *)(j + 112) )
                                  {
                                    v43 = *(_QWORD *)(j + 120);
                                    if ( (*(_BYTE *)(v43 + 140) & 0xFB) == 8
                                      && (sub_8D4C10(v43, dword_4F077C4 != 2) & 1) != 0 )
                                    {
                                      v44 = sub_8D4130(*(_QWORD *)(j + 120));
                                      if ( sub_8D3A70(v44) )
                                        goto LABEL_69;
                                    }
                                  }
                                }
                              }
                            }
                            else
                            {
LABEL_69:
                              v7 = 0;
                            }
                            if ( v71 != (unsigned __int64 *)v70 )
                              sub_823A00((__int64)v71, 8 * v72, v18, v2, v3, v4);
                            if ( v65 != (unsigned __int64 *)v64 )
                              sub_823A00((__int64)v65, 8 * v66, v18, v2, v3, v4);
                            return v7;
                          }
                        }
                      }
                      if ( v26 )
                      {
                        v25 = *(_QWORD *)(v25 + 8);
                        if ( v25 )
                          continue;
                      }
                      v18 = v73;
                      v17 = v67;
                      goto LABEL_66;
                    }
                  }
                  v17 = v67;
LABEL_151:
                  v18 = v73;
                  goto LABEL_66;
                }
LABEL_111:
                v26 = 0;
                if ( *(_BYTE *)(v25 + 80) == 17 )
                {
                  v25 = *(_QWORD *)(v25 + 88);
                  v26 = 1;
                }
                if ( v17 <= 0 )
                  goto LABEL_52;
                goto LABEL_51;
              }
              if ( v73 )
                goto LABEL_80;
LABEL_109:
              v2 = v53;
              v25 = *(_QWORD *)(v53 + 32);
              if ( !v25 )
              {
                v18 = 0;
                goto LABEL_68;
              }
              v18 = 0;
              goto LABEL_111;
            }
LABEL_108:
            v7 = 0;
            v17 = 0;
            goto LABEL_109;
          }
          if ( v11 )
            goto LABEL_122;
        }
        else if ( dword_4F077C4 != 2 )
        {
          goto LABEL_108;
        }
        if ( (*(_BYTE *)(v53 + 177) & 0x40) == 0 )
        {
          v17 = v67;
          v7 = 0;
          goto LABEL_109;
        }
        v47 = *(_QWORD *)(v5 + 160);
        if ( !v47 )
        {
          v17 = v67;
          v7 = 1;
          goto LABEL_109;
        }
        do
        {
          v48 = sub_8D4130(*(_QWORD *)(v47 + 120));
          for ( k = *(_BYTE *)(v48 + 140); k == 12; k = *(_BYTE *)(v48 + 140) )
            v48 = *(_QWORD *)(v48 + 160);
          if ( (unsigned __int8)(k - 9) <= 2u )
          {
            v7 = sub_8E3AD0(v48);
            if ( !v7 )
              goto LABEL_36;
          }
          v47 = *(_QWORD *)(v47 + 112);
        }
        while ( v47 );
        v7 = 1;
        goto LABEL_36;
      }
    }
    return (unsigned __int8)(i - 15) <= 1u;
  }
  return v7;
}
