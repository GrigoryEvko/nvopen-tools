// Function: sub_8CDA30
// Address: 0x8cda30
//
__int64 __fastcall sub_8CDA30(__int64 a1)
{
  unsigned int v1; // r15d
  __int64 *v2; // rax
  __int64 v3; // r12
  __int64 v4; // r13
  _BOOL4 v5; // eax
  __int64 v6; // rdi
  __int64 i; // rbx
  __int64 v8; // rsi
  __int64 j; // r14
  unsigned __int8 v11; // al
  unsigned __int8 v12; // dl
  char v13; // dl
  char v14; // al
  char v15; // di
  unsigned __int8 v16; // al
  __int64 v17; // rcx
  char v18; // r10
  __int64 v19; // rdx
  __int64 v20; // rsi
  char v21; // r9
  unsigned __int8 v22; // r11
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rsi
  char v26; // dl
  __int64 v27; // rcx
  __int64 v28; // rsi
  char v29; // al
  __int64 v30; // rax
  __int64 v31; // rdx
  _QWORD *v32; // rbx
  _QWORD *v33; // r14
  __int64 v34; // rdx
  __int64 *v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 **v39; // rcx
  _QWORD *v40; // rcx
  unsigned __int64 v41; // rdx
  __int64 *v42; // rsi
  bool v43; // sf
  __int64 v44; // rdx
  __int64 v45; // r8
  char v46; // [rsp+Fh] [rbp-51h]
  __int64 *v47; // [rsp+10h] [rbp-50h]
  __int64 **v48; // [rsp+18h] [rbp-48h]
  _QWORD *v49; // [rsp+20h] [rbp-40h]
  __int64 v50; // [rsp+20h] [rbp-40h]
  unsigned __int64 *v51; // [rsp+28h] [rbp-38h]
  __int64 v52; // [rsp+28h] [rbp-38h]

  v1 = 1;
  v2 = *(__int64 **)(a1 + 32);
  if ( v2 )
  {
    v3 = *v2;
    v4 = a1;
    if ( a1 == *v2 )
    {
      v4 = v2[1];
      if ( !v4 || v3 == v4 )
        return 1;
    }
    v5 = sub_8C7610(v4);
    v6 = *(_QWORD *)(v4 + 152);
    v1 = v5;
    for ( i = v6; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v8 = *(_QWORD *)(v3 + 152);
    for ( j = v8; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    if ( v5 )
    {
      if ( !(unsigned int)sub_8DED30(v6, v8, 1048837) )
        goto LABEL_77;
      v11 = *(_BYTE *)(v4 + 193);
      v12 = *(_BYTE *)(v3 + 193);
      if ( ((v12 ^ v11) & 0x10) == 0 )
      {
LABEL_21:
        if ( ((v12 | v11) & 0x10) == 0
          && (*(_BYTE *)(v4 + 206) & 8) == 0
          && (*(_BYTE *)(v3 + 206) & 8) == 0
          && !(unsigned int)sub_8DBAE0(*(_QWORD *)(v4 + 152), *(_QWORD *)(v3 + 152))
          && (*(_BYTE *)(*(_QWORD *)(v4 + 152) + 140LL) != 7 || !sub_8C7240((_QWORD *)v4))
          && (*(_BYTE *)(*(_QWORD *)(v3 + 152) + 140LL) != 7 || !sub_8C7240((_QWORD *)v3)) )
        {
          goto LABEL_77;
        }
        v13 = *(_BYTE *)(v4 + 192);
        v14 = v13 ^ *(_BYTE *)(v3 + 192);
        if ( (v14 & 0xA) != 0
          || (*(_BYTE *)(v4 + 203) & 1) == 0
          && (*(_BYTE *)(v3 + 203) & 1) == 0
          && ((*(_BYTE *)(v3 + 206) ^ *(_BYTE *)(v4 + 206)) & 0x18) != 0 )
        {
          goto LABEL_77;
        }
        if ( dword_4F077C4 == 2 && v14 < 0 )
        {
          v43 = v13 < 0;
          v44 = v3;
          v45 = v4;
          if ( !v43 )
          {
            v44 = v4;
            v45 = v3;
          }
          if ( (*(_BYTE *)(v44 + 195) & 3) != 1
            || (*(_DWORD *)(v44 + 192) & 0x8002000) == 0x8002000
            || *(_DWORD *)(v44 + 160) )
          {
            v50 = v45;
            v52 = v44;
            if ( sub_860410(v44)
              || (*(_WORD *)(v52 + 192) & 0x4003) != 0
              || !sub_860410(v50) && (*(_BYTE *)(v52 + 89) & 4) != 0 )
            {
              goto LABEL_77;
            }
          }
        }
        if ( *(_QWORD *)(v4 + 240) && ((*(_BYTE *)(v3 + 195) ^ *(_BYTE *)(v4 + 195)) & 2) != 0 )
          goto LABEL_77;
        v15 = *(_BYTE *)(v3 + 193);
        if ( ((v15 ^ *(_BYTE *)(v4 + 193)) & 0x80u) != 0 )
          goto LABEL_77;
        v16 = *(_BYTE *)(v4 + 195);
        if ( ((*(_BYTE *)(v3 + 195) ^ v16) & 0x10) != 0
          && ((v16 & 0x10) != 0 && (v15 & 0x20) != 0
           || (*(_BYTE *)(v3 + 195) & 0x10) != 0 && (*(_BYTE *)(v4 + 193) & 0x20) != 0) )
        {
          goto LABEL_77;
        }
        v17 = *(_QWORD *)(v4 + 152);
        v18 = *(_BYTE *)(v17 + 140);
        v19 = v17;
        if ( v18 == 12 )
        {
          do
            v19 = *(_QWORD *)(v19 + 160);
          while ( *(_BYTE *)(v19 + 140) == 12 );
        }
        v20 = *(_QWORD *)(v3 + 152);
        v21 = *(_BYTE *)(v20 + 140);
        v22 = *(_BYTE *)(*(_QWORD *)(v19 + 168) + 20LL);
        v23 = v20;
        if ( v21 == 12 )
        {
          do
            v23 = *(_QWORD *)(v23 + 160);
          while ( *(_BYTE *)(v23 + 140) == 12 );
        }
        if ( ((*(_BYTE *)(*(_QWORD *)(v23 + 168) + 20LL) ^ v22) & 1) == 0 )
          goto LABEL_76;
        if ( v18 == 12 )
        {
          do
            v17 = *(_QWORD *)(v17 + 160);
          while ( *(_BYTE *)(v17 + 140) == 12 );
        }
        if ( (*(_BYTE *)(*(_QWORD *)(v17 + 168) + 20LL) & 1) != 0 && (v15 & 0x20) != 0 )
          goto LABEL_77;
        if ( v21 == 12 )
        {
          do
            v20 = *(_QWORD *)(v20 + 160);
          while ( *(_BYTE *)(v20 + 140) == 12 );
        }
        if ( (*(_BYTE *)(*(_QWORD *)(v20 + 168) + 20LL) & 1) != 0 )
        {
          if ( (*(_BYTE *)(v4 + 193) & 0x20) != 0 )
            goto LABEL_77;
        }
        else
        {
LABEL_76:
          if ( (*(_BYTE *)(v4 + 193) & 0x20) != 0
            && (*(_BYTE *)(v3 + 193) & 0x20) != 0
            && ((*(_BYTE *)(v3 + 204) ^ *(_BYTE *)(v4 + 204)) & 0x7E) != 0 )
          {
            goto LABEL_77;
          }
        }
        if ( ((*(_BYTE *)(v3 + 88) ^ *(_BYTE *)(v4 + 88)) & 3) == 0 )
        {
          if ( ((*(_BYTE *)(v3 + 88) ^ *(_BYTE *)(v4 + 88)) & 0x70) == 0
            || !dword_4D04824
            && (*(char *)(v4 + 192) < 0 ? (v24 = v4, v25 = v3) : (v24 = v3, v25 = v4),
                (*(_WORD *)(v24 + 192) & 0x2080) == 0x2080
             && (*(_BYTE *)(v25 + 193) & 0x20) == 0
             && (*(_BYTE *)(v24 + 88) & 0x70) == 0x10) )
          {
            if ( (v16 & 1) != 0 && (*(char *)(v4 + 192) < 0 || *(char *)(v3 + 192) < 0) && (v16 & 2) == 0 )
            {
              v37 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v4 + 248) + 200LL) + 208LL);
              v38 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 248) + 200LL) + 208LL);
              if ( v37 )
              {
                if ( v38 && *(_DWORD *)(v37 + 124) != *(_DWORD *)(v38 + 124) && !unk_4D04270 )
                {
                  sub_8C6700((__int64 *)v4, (unsigned int *)(v3 + 64), 0x42Au, 0x425u);
                  sub_8C7090(11, v4);
                }
              }
            }
            if ( !*(_QWORD *)(v4 + 256) )
            {
              if ( !*(_QWORD *)(v3 + 256) )
                goto LABEL_62;
              sub_726210(v4);
            }
            if ( !*(_QWORD *)(v3 + 256) )
              sub_726210(v3);
LABEL_62:
            v26 = *(_BYTE *)(v4 + 192);
            if ( unk_4D04958 )
            {
              v27 = *(_QWORD *)(v4 + 152);
              v28 = *(_QWORD *)(v3 + 152);
              if ( v26 >= 0 )
              {
LABEL_64:
                v29 = *(_BYTE *)(v4 + 195);
                goto LABEL_65;
              }
            }
            else
            {
              if ( v26 >= 0 )
              {
                v29 = *(_BYTE *)(v4 + 195);
                if ( ((v29 & 8) == 0 || (v34 = *(_QWORD *)(v4 + 248)) != 0 && (*(_BYTE *)(v34 + 121) & 1) != 0)
                  && (*(_BYTE *)(v4 + 200) & 0x20) == 0
                  && (*(_BYTE *)(v3 + 200) & 0x20) == 0
                  && (*(_BYTE *)(v4 + 193) & 0x20) != 0
                  && (*(_BYTE *)(v3 + 193) & 0x20) != 0 )
                {
                  v35 = *(__int64 **)(v4 + 32);
                  v36 = v4;
                  if ( v35 )
                    v36 = *v35;
                  sub_8C6700((__int64 *)v4, (unsigned int *)(v36 + 64), 0x433u, 0x434u);
                  goto LABEL_69;
                }
                v27 = *(_QWORD *)(v4 + 152);
                v28 = *(_QWORD *)(v3 + 152);
LABEL_65:
                if ( (v29 & 1) == 0 || (*(_BYTE *)(v3 + 195) & 1) == 0 )
                {
                  if ( (*(_BYTE *)(v4 + 89) & 4) == 0 )
                  {
LABEL_69:
                    v30 = *(_QWORD *)(i + 168);
                    if ( (*(_BYTE *)(v30 + 16) & 2) != 0 )
                    {
                      v31 = *(_QWORD *)(j + 168);
                      if ( (*(_BYTE *)(v31 + 16) & 2) != 0 )
                      {
                        v32 = *(_QWORD **)v30;
                        v33 = *(_QWORD **)v31;
                        sub_8C6CA0(v4, v3, 0xBu, (_QWORD *)(v3 + 64));
                        sub_8C6CA0(v3, v4, 0xBu, (_QWORD *)(v4 + 64));
                        for ( ; v32; v33 = (_QWORD *)*v33 )
                        {
                          sub_8C6CA0((__int64)v32, (__int64)v33, 3u, (_QWORD *)(v3 + 64));
                          sub_8C6CA0((__int64)v33, (__int64)v32, 3u, (_QWORD *)(v4 + 64));
                          v32 = (_QWORD *)*v32;
                        }
                      }
                    }
                    return v1;
                  }
                  v46 = 0;
LABEL_68:
                  if ( *(_BYTE *)(v27 + 140) == 7 && *(_BYTE *)(v28 + 140) == 7 )
                  {
                    v39 = **(__int64 ****)(v27 + 168);
                    v48 = v39;
                    v47 = **(__int64 ***)(v28 + 168);
                    if ( v47 )
                    {
                      if ( v39 )
                      {
                        do
                        {
                          if ( v46 || ((_BYTE)v48[4] & 8) != 0 || (v47[4] & 8) != 0 )
                          {
                            v40 = (_QWORD *)v47[7];
                            v41 = (unsigned __int64)v48[7];
                            if ( v40 && v41 )
                            {
                              do
                              {
                                v49 = v40;
                                v51 = (unsigned __int64 *)v41;
                                sub_8CD5A0(*(__int64 **)(v41 + 16));
                                v41 = *v51;
                                v40 = (_QWORD *)*v49;
                              }
                              while ( *v51 && v40 );
                            }
                            if ( __PAIR128__((unsigned __int64)v40, v41) != 0 )
                              sub_8C6700((__int64 *)v4, (unsigned int *)(v3 + 64), 0x701u, 0x700u);
                          }
                          v42 = (__int64 *)*v47;
                          v48 = (__int64 **)*v48;
                          v47 = (__int64 *)*v47;
                        }
                        while ( v48 && v42 );
                      }
                    }
                  }
                  goto LABEL_69;
                }
LABEL_67:
                v46 = 1;
                goto LABEL_68;
              }
              v27 = *(_QWORD *)(v4 + 152);
              v28 = *(_QWORD *)(v3 + 152);
            }
            if ( *(char *)(v3 + 192) < 0 )
              goto LABEL_67;
            goto LABEL_64;
          }
        }
LABEL_77:
        v1 = 0;
        sub_8C6700((__int64 *)v4, (unsigned int *)(v3 + 64), 0x42Au, 0x425u);
        sub_8C7090(11, v4);
        return v1;
      }
      if ( (v11 & 0x10) != 0 )
      {
        if ( !*(_BYTE *)(v4 + 174) && dword_4F077C4 != 2 && !*(_WORD *)(v4 + 176) )
          goto LABEL_21;
        if ( (v12 & 0x10) == 0 )
        {
          if ( *(_BYTE *)(v4 + 174) != 5 || (unsigned __int8)(*(_BYTE *)(v4 + 176) - 1) > 3u )
            goto LABEL_77;
          goto LABEL_21;
        }
        if ( *(_BYTE *)(v3 + 174) )
          goto LABEL_18;
      }
      else
      {
        if ( (v12 & 0x10) == 0 )
          goto LABEL_77;
        if ( *(_BYTE *)(v3 + 174) )
          goto LABEL_19;
      }
      if ( dword_4F077C4 != 2 && !*(_WORD *)(v3 + 176) )
        goto LABEL_21;
      if ( (v11 & 0x10) == 0 )
        goto LABEL_77;
LABEL_18:
      if ( *(_BYTE *)(v4 + 174) == 5 && (unsigned __int8)(*(_BYTE *)(v4 + 176) - 1) <= 3u )
        goto LABEL_21;
LABEL_19:
      if ( *(_BYTE *)(v3 + 174) != 5 || (unsigned __int8)(*(_BYTE *)(v3 + 176) - 1) > 3u )
        goto LABEL_77;
      goto LABEL_21;
    }
  }
  return v1;
}
