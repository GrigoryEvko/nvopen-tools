// Function: sub_13047E0
// Address: 0x13047e0
//
__int64 __fastcall sub_13047E0(unsigned __int64 a1)
{
  __int64 v2; // r12
  unsigned int v3; // r14d
  __int64 v4; // rdx
  __int64 v5; // r10
  unsigned __int64 v6; // r15
  __int64 v7; // rbx
  __int64 *v8; // rax
  __int64 v9; // r8
  _QWORD *v10; // rsi
  __int64 v11; // rdx
  __int64 v13; // rax
  __int64 v14; // r13
  int *v15; // rax
  __int64 v16; // rdi
  unsigned __int8 v17; // r15
  __int64 v18; // rcx
  __int64 v19; // r14
  size_t v20; // r10
  unsigned int v21; // r9d
  __int64 v22; // rbx
  void **v23; // rax
  void *v24; // r8
  void **v25; // rsi
  __int64 v26; // rdx
  void *v27; // rdx
  char v28; // cl
  int v29; // edx
  __int64 v30; // rbx
  char v31; // cl
  unsigned __int64 v32; // rax
  __int64 v33; // rax
  __int64 *v34; // rdx
  _QWORD *v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rsi
  int *v38; // rax
  __int64 v39; // rax
  char v40; // cl
  __int64 v41; // rax
  __int64 v42; // rbx
  void **v43; // rax
  void **v44; // rsi
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  void *v48; // rax
  __int64 v49; // rax
  void *v50; // rax
  __int64 v51; // rdi
  __int64 v52; // r14
  __int64 v53; // rax
  __int64 v54; // [rsp+0h] [rbp-90h]
  __int64 v55; // [rsp+8h] [rbp-88h]
  __int64 v56; // [rsp+10h] [rbp-80h]
  unsigned int v57; // [rsp+10h] [rbp-80h]
  __int64 v58; // [rsp+18h] [rbp-78h]
  void *v59; // [rsp+18h] [rbp-78h]
  size_t v60; // [rsp+18h] [rbp-78h]
  __int64 v61; // [rsp+20h] [rbp-70h]
  __int64 v62; // [rsp+20h] [rbp-70h]
  size_t v63; // [rsp+20h] [rbp-70h]
  size_t v64; // [rsp+20h] [rbp-70h]
  __int64 v65; // [rsp+20h] [rbp-70h]
  unsigned int v66; // [rsp+20h] [rbp-70h]
  void *v67; // [rsp+20h] [rbp-70h]
  size_t v68; // [rsp+20h] [rbp-70h]
  __int64 v69; // [rsp+28h] [rbp-68h]
  void *v70; // [rsp+28h] [rbp-68h]
  __int64 v71; // [rsp+28h] [rbp-68h]
  unsigned int v72; // [rsp+28h] [rbp-68h]
  size_t v73; // [rsp+28h] [rbp-68h]
  __int64 v74; // [rsp+28h] [rbp-68h]
  size_t v75; // [rsp+28h] [rbp-68h]
  void *v76; // [rsp+28h] [rbp-68h]
  size_t v77; // [rsp+28h] [rbp-68h]
  unsigned int v78; // [rsp+28h] [rbp-68h]
  unsigned __int64 v79; // [rsp+30h] [rbp-60h] BYREF
  __int128 v80; // [rsp+38h] [rbp-58h]
  __int64 v81; // [rsp+48h] [rbp-48h]
  __int64 v82; // [rsp+50h] [rbp-40h]

  v2 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v16 = v2;
    v2 = sub_1313D30(v2, 0);
    if ( *(_BYTE *)(v2 + 816) )
    {
      if ( dword_4C6F034[0] && (unsigned __int8)sub_13022D0(v16, 0) )
        goto LABEL_16;
      v17 = unk_4F96994;
      if ( a1 > 0x1000 )
      {
        if ( a1 > 0x7000000000000000LL )
          goto LABEL_55;
        v40 = 7;
        _BitScanReverse64((unsigned __int64 *)&v41, 2 * a1 - 1);
        if ( (unsigned int)v41 >= 7 )
          v40 = v41;
        if ( (unsigned int)v41 < 6 )
          LODWORD(v41) = 6;
        v18 = ((unsigned int)(((a1 - 1) & (-1LL << (v40 - 3))) >> (v40 - 3)) & 3) + 4 * (_DWORD)v41 - 23;
      }
      else
      {
        v18 = byte_5060800[(a1 + 7) >> 3];
      }
      if ( (unsigned int)v18 > 0xE7 )
        goto LABEL_55;
      v19 = (unsigned int)v18;
      v20 = qword_505FA40[(unsigned int)v18];
      if ( *(char *)(v2 + 1) > 0 )
      {
        v37 = qword_50579C0[0];
        if ( !qword_50579C0[0] )
        {
          v63 = qword_505FA40[(unsigned int)v18];
          v72 = v18;
          v36 = sub_1300B80(v2, 0, (__int64)&off_49E8000);
          v18 = v72;
          v20 = v63;
          v37 = v36;
          if ( !v36 && !unk_505F9B8 )
            goto LABEL_55;
        }
        v21 = v17;
      }
      else
      {
        v21 = unk_4F96994;
        if ( *(_BYTE *)v2 )
        {
          if ( a1 <= 0x3800 )
          {
            v22 = v2 + 24LL * (unsigned int)v18;
            v23 = *(void ***)(v22 + 864);
            v24 = *v23;
            v25 = v23 + 1;
            if ( (_WORD)v23 == *(_WORD *)(v22 + 880) )
            {
              if ( (_WORD)v23 == *(_WORD *)(v22 + 884) )
              {
                v54 = 24LL * (unsigned int)v18;
                v60 = qword_505FA40[(unsigned int)v18];
                v66 = v18;
                v46 = sub_1302E60(v2, 0);
                v74 = v46;
                if ( !v46 )
                  goto LABEL_55;
                if ( !*(_WORD *)(unk_5060A20 + 2 * v19) )
                {
                  v49 = sub_1317CF0(v2, v46, a1, v66, v17);
                  v20 = v60;
                  v24 = (void *)v49;
                  goto LABEL_30;
                }
                v56 = v2 + 856 + v54 + 8;
                sub_1310140(v2, v2 + 856, v56, v66, 1);
                v47 = sub_13100A0(v2, v74, v2 + 856, v56, v66);
                v20 = v60;
                v24 = (void *)v47;
                if ( !(_BYTE)v79 )
                  goto LABEL_55;
              }
              else
              {
                *(_QWORD *)(v22 + 864) = v25;
                *(_WORD *)(v22 + 880) = (_WORD)v25;
              }
            }
            else
            {
              *(_QWORD *)(v22 + 864) = v25;
            }
            if ( v17 )
            {
              v75 = v20;
              v48 = memset(v24, 0, qword_505FA40[v19]);
              v20 = v75;
              v24 = v48;
            }
            ++*(_QWORD *)(v22 + 872);
LABEL_30:
            if ( v24 )
            {
LABEL_31:
              LOBYTE(v79) = 1;
              *(_QWORD *)&v80 = v2 + 824;
              *((_QWORD *)&v80 + 1) = v2 + 8;
              v81 = v2 + 16;
              v82 = v2 + 832;
              v26 = *(_QWORD *)(v2 + 824);
              *(_QWORD *)(v2 + 824) = v20 + v26;
              if ( v20 >= *(_QWORD *)(v2 + 16) - v26 )
              {
                v59 = v24;
                v64 = v20;
                sub_13133F0(v2, &v79);
                v24 = v59;
                v20 = v64;
              }
              v27 = v24;
              if ( !v17 && unk_4F969A2 )
              {
                v67 = v24;
                v76 = v24;
                off_4C6F0B8(v24, v20);
                v24 = v76;
                v27 = v67;
              }
              goto LABEL_36;
            }
            goto LABEL_55;
          }
          if ( a1 <= unk_5060A10 )
          {
            v42 = v2 + 24LL * (unsigned int)v18;
            v43 = *(void ***)(v42 + 864);
            v24 = *v43;
            v44 = v43 + 1;
            if ( (_WORD)v43 != *(_WORD *)(v42 + 880) )
            {
              *(_QWORD *)(v42 + 864) = v44;
LABEL_72:
              if ( v17 )
              {
                v77 = v20;
                v50 = memset(v24, 0, qword_505FA40[(unsigned int)v18]);
                v20 = v77;
                v24 = v50;
              }
              ++*(_QWORD *)(v42 + 872);
              goto LABEL_30;
            }
            v51 = v2 + 24LL * (unsigned int)v18;
            if ( (_WORD)v43 != *(_WORD *)(v51 + 884) )
            {
              *(_QWORD *)(v42 + 864) = v44;
              *(_WORD *)(v51 + 880) = (_WORD)v44;
              goto LABEL_72;
            }
            v55 = 24LL * (unsigned int)v18;
            v57 = unk_4F96994;
            v68 = qword_505FA40[(unsigned int)v18];
            v78 = v18;
            v52 = sub_1303050(v2, 0);
            if ( v52 )
            {
              sub_1310140(v2, v2 + 856, v2 + 856 + v55 + 8, v78, 0);
              _BitScanReverse64((unsigned __int64 *)&v53, 2 * a1 - 1);
              if ( (unsigned __int64)(int)v53 < 7 )
                LOBYTE(v53) = 7;
              v24 = (void *)sub_1309DC0(
                              v2,
                              v52,
                              -(1LL << ((unsigned __int8)v53 - 3)) & ((1LL << ((unsigned __int8)v53 - 3)) + a1 - 1),
                              v57);
              if ( v24 )
              {
                v20 = v68;
                goto LABEL_31;
              }
            }
LABEL_55:
            v38 = __errno_location();
            v27 = 0;
            v24 = 0;
            *v38 = 12;
LABEL_36:
            v70 = v24;
            v79 = a1;
            v80 = 0;
            sub_1346E80(0, v24, v27, &v79);
            return (__int64)v70;
          }
        }
        v37 = 0;
      }
      v73 = v20;
      v39 = sub_1317CF0(v2, v37, a1, v18, v21);
      v20 = v73;
      v24 = (void *)v39;
      goto LABEL_30;
    }
  }
  if ( a1 <= 0x1000 )
  {
    v3 = byte_5060800[(a1 + 7) >> 3];
    if ( v3 > 0xE7 )
      goto LABEL_16;
    v4 = byte_5060800[(a1 + 7) >> 3];
    v5 = v2 + 856;
    v6 = qword_505FA40[v4];
    goto LABEL_5;
  }
  if ( a1 > 0x7000000000000000LL )
    goto LABEL_16;
  v28 = 7;
  v29 = 6;
  _BitScanReverse64((unsigned __int64 *)&v30, 2 * a1 - 1);
  if ( (unsigned int)v30 >= 7 )
    v28 = v30;
  v31 = v28 - 3;
  if ( (unsigned int)v30 >= 6 )
    v29 = v30;
  v32 = (((a1 - 1) & (-1LL << v31)) >> v31) & 3;
  v3 = v32 + 4 * v29 - 23;
  if ( (_DWORD)v32 + 4 * v29 == 255 )
    goto LABEL_16;
  v4 = v3;
  v5 = v2 + 856;
  v6 = qword_505FA40[v3];
  if ( a1 <= 0x3800 )
  {
LABEL_5:
    v7 = v2 + 24 * v4;
    v8 = *(__int64 **)(v7 + 864);
    v9 = *v8;
    v10 = v8 + 1;
    if ( (_WORD)v8 != *(_WORD *)(v7 + 880) )
    {
      *(_QWORD *)(v7 + 864) = v10;
LABEL_7:
      ++*(_QWORD *)(v7 + 872);
      goto LABEL_8;
    }
    if ( (_WORD)v8 != *(_WORD *)(v7 + 884) )
    {
      *(_QWORD *)(v7 + 864) = v10;
      *(_WORD *)(v7 + 880) = (_WORD)v10;
      goto LABEL_7;
    }
    v58 = v4;
    v61 = 24 * v4;
    v69 = v5;
    v13 = sub_1302E60(v2, 0);
    if ( !v13 )
      goto LABEL_16;
    if ( *(_WORD *)(unk_5060A20 + 2 * v58) )
    {
      v14 = v69 + v61 + 8;
      v62 = v13;
      sub_1310140(v2, v69, v14, v3, 1);
      v9 = sub_13100A0(v2, v62, v69, v14, v3);
      if ( !(_BYTE)v79 )
        goto LABEL_16;
      goto LABEL_7;
    }
    v9 = sub_1317CF0(v2, v13, a1, v3, 0);
LABEL_8:
    if ( v9 )
      goto LABEL_9;
LABEL_16:
    v15 = __errno_location();
    v9 = 0;
    *v15 = 12;
    return v9;
  }
  if ( a1 > unk_5060A10 )
  {
    v9 = sub_1317CF0(v2, 0, a1, v3, 0);
    goto LABEL_8;
  }
  v33 = v2 + 24LL * v3;
  v34 = *(__int64 **)(v33 + 864);
  v9 = *v34;
  v35 = v34 + 1;
  if ( (_WORD)v34 != *(_WORD *)(v33 + 880) )
  {
    *(_QWORD *)(v33 + 864) = v35;
LABEL_47:
    ++*(_QWORD *)(v33 + 872);
    goto LABEL_8;
  }
  if ( (_WORD)v34 != *(_WORD *)(v33 + 884) )
  {
    *(_QWORD *)(v33 + 864) = v35;
    *(_WORD *)(v33 + 880) = (_WORD)v35;
    goto LABEL_47;
  }
  v45 = sub_1302E60(v2, 0);
  if ( !v45 )
    goto LABEL_16;
  v65 = v45;
  sub_1310140(v2, v2 + 856, v2 + 856 + 24LL * v3 + 8, v3, 0);
  if ( (unsigned __int64)(int)v30 < 7 )
    LOBYTE(v30) = 7;
  v9 = sub_1309DC0(v2, v65, -(1LL << ((unsigned __int8)v30 - 3)) & ((1LL << ((unsigned __int8)v30 - 3)) + a1 - 1), 0);
  if ( !v9 )
    goto LABEL_16;
LABEL_9:
  LOBYTE(v79) = 1;
  *(_QWORD *)&v80 = v2 + 824;
  *((_QWORD *)&v80 + 1) = v2 + 8;
  v81 = v2 + 16;
  v82 = v2 + 832;
  v11 = *(_QWORD *)(v2 + 824);
  *(_QWORD *)(v2 + 824) = v11 + v6;
  if ( *(_QWORD *)(v2 + 16) - v11 <= v6 )
  {
    v71 = v9;
    sub_13133F0(v2, &v79);
    return v71;
  }
  return v9;
}
