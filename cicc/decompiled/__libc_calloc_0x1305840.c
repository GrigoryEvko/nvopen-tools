// Function: __libc_calloc
// Address: 0x1305840
//
// Alternative name is 'calloc'
void *__fastcall _libc_calloc(__int64 a1, unsigned __int64 a2)
{
  __int64 v4; // r12
  unsigned __int64 v5; // r15
  __int64 v6; // rcx
  __int64 v7; // rbx
  unsigned __int64 v8; // r10
  __int64 v9; // r13
  void **v10; // rax
  void *v11; // r8
  void **v12; // rsi
  void *v13; // rax
  unsigned __int64 v14; // r10
  __int64 v15; // r8
  __int64 v16; // rdx
  __int64 v18; // rax
  int *v19; // rax
  __int64 v20; // rdi
  unsigned __int64 v21; // r9
  __int64 v22; // rcx
  __int64 v23; // r15
  __int64 v24; // r10
  void **v25; // rax
  void *v26; // r8
  void **v27; // rsi
  void *v28; // r8
  __int64 v29; // rdx
  void *v30; // rdx
  char v31; // cl
  __int64 v32; // rax
  void **v33; // rax
  __int64 v34; // rax
  __int64 v35; // r15
  __int64 v36; // rax
  __int64 v37; // rsi
  int *v38; // rax
  __int64 v39; // r10
  void **v40; // rax
  void *v41; // r8
  void **v42; // rsi
  __int64 v43; // rax
  char v44; // cl
  __int64 v45; // rax
  __int64 v46; // r13
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rcx
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rdi
  __int64 v54; // r14
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // [rsp+8h] [rbp-98h]
  unsigned int v58; // [rsp+10h] [rbp-90h]
  unsigned int v59; // [rsp+18h] [rbp-88h]
  __int64 v60; // [rsp+18h] [rbp-88h]
  __int64 v61; // [rsp+18h] [rbp-88h]
  __int64 v62; // [rsp+20h] [rbp-80h]
  __int64 v63; // [rsp+20h] [rbp-80h]
  __int64 v64; // [rsp+20h] [rbp-80h]
  unsigned int v65; // [rsp+28h] [rbp-78h]
  unsigned int v66; // [rsp+28h] [rbp-78h]
  __int64 v67; // [rsp+30h] [rbp-70h]
  unsigned __int64 v68; // [rsp+30h] [rbp-70h]
  __int64 v69; // [rsp+30h] [rbp-70h]
  void *v70; // [rsp+30h] [rbp-70h]
  unsigned __int64 v71; // [rsp+30h] [rbp-70h]
  __int64 v72; // [rsp+30h] [rbp-70h]
  unsigned __int64 v73; // [rsp+30h] [rbp-70h]
  unsigned __int64 v74; // [rsp+38h] [rbp-68h]
  unsigned __int64 v75; // [rsp+38h] [rbp-68h]
  unsigned __int64 v76; // [rsp+38h] [rbp-68h]
  void *v77; // [rsp+38h] [rbp-68h]
  unsigned int v78; // [rsp+38h] [rbp-68h]
  __int64 v79; // [rsp+38h] [rbp-68h]
  unsigned int v80; // [rsp+38h] [rbp-68h]
  __int64 v81; // [rsp+40h] [rbp-60h] BYREF
  __int64 v82; // [rsp+48h] [rbp-58h]
  __int64 v83; // [rsp+50h] [rbp-50h]
  __int64 v84; // [rsp+58h] [rbp-48h]
  __int64 v85; // [rsp+60h] [rbp-40h]

  v4 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v20 = v4;
    v4 = sub_1313D30(v4, 0);
    if ( *(_BYTE *)(v4 + 816) )
    {
      if ( dword_4C6F034[0] && (unsigned __int8)sub_13022D0(v20, 0) )
        goto LABEL_17;
      v21 = a2 * a1;
      if ( a2 * a1 )
      {
        if ( ((a2 | a1) & 0xFFFFFFFF00000000LL) != 0 && a1 != v21 / a2 )
          goto LABEL_57;
        if ( v21 > 0x1000 )
        {
          if ( v21 > 0x7000000000000000LL )
            goto LABEL_57;
          v44 = 7;
          _BitScanReverse64((unsigned __int64 *)&v45, 2 * v21 - 1);
          if ( (unsigned int)v45 >= 7 )
            v44 = v45;
          if ( (unsigned int)v45 < 6 )
            LODWORD(v45) = 6;
          v22 = ((unsigned int)(((v21 - 1) & (-1LL << (v44 - 3))) >> (v44 - 3)) & 3) + 4 * (_DWORD)v45 - 23;
LABEL_24:
          if ( (unsigned int)v22 > 0xE7 )
            goto LABEL_57;
          v23 = (unsigned int)v22;
          v76 = qword_505FA40[(unsigned int)v22];
          if ( *(char *)(v4 + 1) > 0 )
          {
            v37 = qword_50579C0[0];
            if ( qword_50579C0[0] )
              goto LABEL_55;
            v65 = v22;
            v43 = sub_1300B80(v4, 0, (__int64)&off_49E8000);
            v21 = a2 * a1;
            v22 = v65;
            v37 = v43;
            if ( v43 )
              goto LABEL_55;
            if ( !unk_505F9B8 )
              goto LABEL_57;
          }
          else if ( *(_BYTE *)v4 )
          {
            if ( v21 <= 0x3800 )
            {
              v24 = v4 + 24LL * (unsigned int)v22;
              v25 = *(void ***)(v24 + 864);
              v26 = *v25;
              v27 = v25 + 1;
              if ( (_WORD)v25 != *(_WORD *)(v24 + 880) )
              {
                *(_QWORD *)(v24 + 864) = v27;
LABEL_30:
                v67 = v24;
                v28 = memset(v26, 0, qword_505FA40[v23]);
                ++*(_QWORD *)(v67 + 872);
                goto LABEL_31;
              }
              if ( (_WORD)v25 != *(_WORD *)(v24 + 884) )
              {
                *(_QWORD *)(v24 + 864) = v27;
                *(_WORD *)(v24 + 880) = (_WORD)v27;
                goto LABEL_30;
              }
              v57 = 24LL * (unsigned int)v22;
              v59 = v22;
              v49 = sub_1302E60(v4, 0);
              v72 = v49;
              v50 = v59;
              if ( !v49 )
              {
LABEL_57:
                v38 = __errno_location();
                v30 = 0;
                v28 = 0;
                *v38 = 12;
LABEL_34:
                v77 = v28;
                v83 = 0;
                v81 = a1;
                v82 = a2;
                sub_1346E80(3, v28, v30, &v81);
                return v77;
              }
              if ( *(_WORD *)(unk_5060A20 + 2 * v23) )
              {
                v58 = v59;
                v60 = v4 + 856 + v57 + 8;
                sub_1310140(v4, v4 + 856, v60, v50, 1);
                v51 = sub_13100A0(v4, v72, v4 + 856, v60, v58);
                v24 = v4 + v57;
                v26 = (void *)v51;
                if ( (_BYTE)v81 )
                  goto LABEL_30;
                goto LABEL_57;
              }
              v28 = (void *)sub_1317CF0(v4, v49, a2 * a1, v59, 1);
LABEL_56:
              if ( v28 )
                goto LABEL_31;
              goto LABEL_57;
            }
            if ( v21 <= unk_5060A10 )
            {
              v39 = v4 + 24LL * (unsigned int)v22;
              v40 = *(void ***)(v39 + 864);
              v41 = *v40;
              v42 = v40 + 1;
              if ( (_WORD)v40 == *(_WORD *)(v39 + 880) )
              {
                v53 = v4 + 24LL * (unsigned int)v22;
                if ( (_WORD)v40 == *(_WORD *)(v53 + 884) )
                {
                  v61 = 24LL * (unsigned int)v22;
                  v66 = v22;
                  v73 = a2 * a1;
                  v54 = sub_1302E60(v4, 0);
                  if ( !v54 )
                    goto LABEL_57;
                  sub_1310140(v4, v4 + 856, v4 + 856 + v61 + 8, v66, 0);
                  if ( v73 > 0x7000000000000000LL )
                  {
                    v56 = 0;
                  }
                  else
                  {
                    _BitScanReverse64((unsigned __int64 *)&v55, 2 * v73 - 1);
                    if ( (unsigned __int64)(int)v55 < 7 )
                      LOBYTE(v55) = 7;
                    v56 = -(1LL << ((unsigned __int8)v55 - 3)) & (v73 + (1LL << ((unsigned __int8)v55 - 3)) - 1);
                  }
                  v28 = (void *)sub_1309DC0(v4, v54, v56, 1);
                  if ( !v28 )
                    goto LABEL_57;
LABEL_31:
                  LOBYTE(v81) = 1;
                  v82 = v4 + 824;
                  v83 = v4 + 8;
                  v84 = v4 + 16;
                  v85 = v4 + 832;
                  v29 = *(_QWORD *)(v4 + 824);
                  *(_QWORD *)(v4 + 824) = v76 + v29;
                  if ( v76 >= *(_QWORD *)(v4 + 16) - v29 )
                  {
                    v70 = v28;
                    sub_13133F0(v4, &v81);
                    v28 = v70;
                  }
                  v30 = v28;
                  goto LABEL_34;
                }
                *(_QWORD *)(v39 + 864) = v42;
                *(_WORD *)(v53 + 880) = (_WORD)v42;
              }
              else
              {
                *(_QWORD *)(v39 + 864) = v42;
              }
              v69 = v4 + 24LL * (unsigned int)v22;
              v28 = memset(v41, 0, qword_505FA40[(unsigned int)v22]);
              ++*(_QWORD *)(v69 + 872);
              goto LABEL_31;
            }
          }
          v37 = 0;
LABEL_55:
          v28 = (void *)sub_1317CF0(v4, v37, v21, v22, 1);
          goto LABEL_56;
        }
      }
      else if ( a2 && a1 )
      {
        goto LABEL_57;
      }
      v22 = byte_5060800[(v21 + 7) >> 3];
      goto LABEL_24;
    }
  }
  v5 = a2 * a1;
  if ( !(a2 * a1) )
  {
    if ( a2 && a1 )
      goto LABEL_17;
LABEL_5:
    v6 = byte_5060800[(v5 + 7) >> 3];
    goto LABEL_6;
  }
  if ( ((a2 | a1) & 0xFFFFFFFF00000000LL) != 0 && a1 != v5 / a2 )
    goto LABEL_17;
  if ( v5 <= 0x1000 )
    goto LABEL_5;
  if ( v5 > 0x7000000000000000LL )
    goto LABEL_17;
  v31 = 7;
  _BitScanReverse64((unsigned __int64 *)&v32, 2 * v5 - 1);
  if ( (unsigned int)v32 >= 7 )
    v31 = v32;
  if ( (unsigned int)v32 < 6 )
    LODWORD(v32) = 6;
  v6 = ((unsigned int)(((v5 - 1) & (-1LL << (v31 - 3))) >> (v31 - 3)) & 3) + 4 * (_DWORD)v32 - 23;
LABEL_6:
  if ( (unsigned int)v6 > 0xE7 )
    goto LABEL_17;
  v7 = (unsigned int)v6;
  v8 = qword_505FA40[(unsigned int)v6];
  if ( v5 <= 0x3800 )
  {
    v9 = v4 + 24LL * (unsigned int)v6;
    v10 = *(void ***)(v9 + 864);
    v11 = *v10;
    v12 = v10 + 1;
    if ( (_WORD)v10 != *(_WORD *)(v9 + 880) )
    {
LABEL_9:
      *(_QWORD *)(v9 + 864) = v12;
LABEL_10:
      v74 = v8;
      v13 = memset(v11, 0, qword_505FA40[v7]);
      ++*(_QWORD *)(v9 + 872);
      v14 = v74;
      v15 = (__int64)v13;
      goto LABEL_11;
    }
    if ( (_WORD)v10 == *(_WORD *)(v9 + 884) )
    {
      v62 = 24LL * (unsigned int)v6;
      v68 = qword_505FA40[(unsigned int)v6];
      v78 = v6;
      v34 = sub_1302E60(v4, 0);
      if ( !v34 )
        goto LABEL_17;
      if ( *(_WORD *)(unk_5060A20 + 2 * v7) )
      {
        v35 = v4 + 856 + v62 + 8;
        v63 = v34;
        sub_1310140(v4, v4 + 856, v35, v78, 1);
        v36 = sub_13100A0(v4, v63, v4 + 856, v35, v78);
        v8 = v68;
        v11 = (void *)v36;
        if ( (_BYTE)v81 )
          goto LABEL_10;
LABEL_17:
        v19 = __errno_location();
        v15 = 0;
        *v19 = 12;
        return (void *)v15;
      }
      v52 = sub_1317CF0(v4, v34, v5, v78, 1);
      v14 = v68;
      v15 = v52;
      goto LABEL_16;
    }
LABEL_48:
    *(_QWORD *)(v9 + 864) = v12;
    *(_WORD *)(v9 + 880) = (_WORD)v12;
    goto LABEL_10;
  }
  if ( v5 <= unk_5060A10 )
  {
    v9 = v4 + 24LL * (unsigned int)v6;
    v33 = *(void ***)(v9 + 864);
    v11 = *v33;
    v12 = v33 + 1;
    if ( (_WORD)v33 != *(_WORD *)(v9 + 880) )
      goto LABEL_9;
    if ( (_WORD)v33 == *(_WORD *)(v9 + 884) )
    {
      v64 = 24LL * (unsigned int)v6;
      v71 = qword_505FA40[(unsigned int)v6];
      v80 = v6;
      v46 = sub_1302E60(v4, 0);
      if ( !v46 )
        goto LABEL_17;
      sub_1310140(v4, v4 + 856, v4 + 856 + v64 + 8, v80, 0);
      _BitScanReverse64((unsigned __int64 *)&v47, 2 * v5 - 1);
      if ( (unsigned __int64)(int)v47 < 7 )
        LOBYTE(v47) = 7;
      v48 = sub_1309DC0(
              v4,
              v46,
              -(1LL << ((unsigned __int8)v47 - 3)) & (v5 + (1LL << ((unsigned __int8)v47 - 3)) - 1),
              1);
      v14 = v71;
      v15 = v48;
      if ( !v48 )
        goto LABEL_17;
      goto LABEL_11;
    }
    goto LABEL_48;
  }
  v75 = qword_505FA40[(unsigned int)v6];
  v18 = sub_1317CF0(v4, 0, v5, v6, 1);
  v14 = v75;
  v15 = v18;
LABEL_16:
  if ( !v15 )
    goto LABEL_17;
LABEL_11:
  LOBYTE(v81) = 1;
  v82 = v4 + 824;
  v83 = v4 + 8;
  v84 = v4 + 16;
  v85 = v4 + 832;
  v16 = *(_QWORD *)(v4 + 824);
  *(_QWORD *)(v4 + 824) = v14 + v16;
  if ( v14 >= *(_QWORD *)(v4 + 16) - v16 )
  {
    v79 = v15;
    sub_13133F0(v4, &v81);
    return (void *)v79;
  }
  return (void *)v15;
}
