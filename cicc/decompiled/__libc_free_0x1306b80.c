// Function: __libc_free
// Address: 0x1306b80
//
// Alternative name is 'free'
void __fastcall _libc_free(unsigned __int64 a1)
{
  _QWORD *v2; // r12
  _QWORD *v3; // r15
  unsigned __int64 v4; // r14
  _QWORD *v5; // rdx
  char *v6; // rbx
  __int64 v7; // rax
  __int64 *v8; // rax
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  size_t v11; // r10
  _QWORD *v12; // r14
  _QWORD *v13; // rbx
  __int64 v14; // rsi
  __int64 v15; // rdx
  _BYTE *v16; // rax
  _BYTE *v17; // r14
  _QWORD *v18; // rdx
  char *v19; // rbx
  unsigned __int64 v20; // r15
  __int64 v21; // rsi
  unsigned __int64 *v22; // rax
  unsigned __int64 v23; // rax
  char v24; // cl
  unsigned __int64 v25; // rax
  __int64 v26; // r9
  _BYTE *v27; // r15
  _BYTE *v28; // rbx
  __int64 v29; // rcx
  __int64 v30; // rcx
  __int64 *v31; // r13
  __int64 v32; // rsi
  __int64 v33; // rcx
  __int64 v34; // rax
  _QWORD *v35; // r14
  __int64 v36; // rcx
  _QWORD *v37; // rcx
  unsigned int i; // esi
  __int64 v39; // rdi
  _QWORD *v40; // rsi
  _QWORD *v41; // rdi
  __int64 v42; // rax
  unsigned __int16 v43; // cx
  __int64 v44; // rax
  __int64 v45; // rcx
  __int64 *v46; // rax
  __int64 k; // rax
  int v48; // ecx
  _QWORD *v49; // rdi
  __int64 v50; // rax
  _QWORD *v51; // rcx
  __int64 v52; // rax
  __int64 j; // rax
  int v54; // edi
  _QWORD *v55; // rsi
  _QWORD *v56; // rdx
  __int64 v57; // rax
  __int64 m; // rax
  int v59; // edi
  _QWORD *v60; // rsi
  _QWORD *v61; // rdx
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // rcx
  unsigned __int64 v64; // rax
  unsigned __int64 v65; // rcx
  unsigned __int64 v66; // rax
  __int64 v67; // rsi
  __int64 v68; // [rsp-B0h] [rbp-B0h]
  char v69; // [rsp-A1h] [rbp-A1h]
  unsigned __int64 v70; // [rsp-A0h] [rbp-A0h]
  _QWORD *v71; // [rsp-98h] [rbp-98h]
  __int64 v72; // [rsp-98h] [rbp-98h]
  size_t v73; // [rsp-90h] [rbp-90h]
  size_t v74; // [rsp-90h] [rbp-90h]
  size_t v75; // [rsp-90h] [rbp-90h]
  size_t v76; // [rsp-90h] [rbp-90h]
  size_t v77; // [rsp-90h] [rbp-90h]
  size_t v78; // [rsp-90h] [rbp-90h]
  size_t v79; // [rsp-90h] [rbp-90h]
  size_t v80; // [rsp-90h] [rbp-90h]
  size_t v81; // [rsp-90h] [rbp-90h]
  size_t v82; // [rsp-90h] [rbp-90h]
  unsigned __int64 v83; // [rsp-88h] [rbp-88h] BYREF
  __int128 v84; // [rsp-80h] [rbp-80h]
  char v85; // [rsp-68h] [rbp-68h] BYREF
  _QWORD *v86; // [rsp-60h] [rbp-60h]
  _QWORD *v87; // [rsp-58h] [rbp-58h]
  _QWORD *v88; // [rsp-50h] [rbp-50h]
  _QWORD *v89; // [rsp-48h] [rbp-48h]

  v62 = __readfsqword(0) + ((a1 >> 26) & 0xF0) - 2664;
  if ( (a1 & 0xFFFFFFFFC0000000LL) == *(_QWORD *)(v62 + 432) )
  {
    v63 = *(_QWORD *)(*(_QWORD *)(v62 + 440) + 8 * ((a1 >> 12) & 0x3FFFF));
    v64 = HIWORD(v63);
    if ( (v63 & 1) != 0 )
    {
      v65 = __readfsqword(0xFFFFF8E0) + qword_505FA40[(unsigned int)v64];
      if ( __readfsqword(0xFFFFF8E8) > v65 )
      {
        v66 = __readfsqword(0) + 24LL * (unsigned int)v64 - 2664;
        v67 = *(_QWORD *)(v66 + 864);
        if ( *(_WORD *)(v66 + 882) != (_WORD)v67 )
        {
          *(_QWORD *)(v66 + 864) = v67 - 8;
          *(_QWORD *)(v67 - 8) = a1;
          __writefsqword(0xFFFFF8E0, v65);
          return;
        }
      }
    }
  }
  if ( !a1 )
  {
    nullsub_2016();
    return;
  }
  v2 = (_QWORD *)(__readfsqword(0) - 2664);
  if ( !__readfsbyte(0xFFFFF8C8) || (v16 = (_BYTE *)sub_1313D30(v2, 1), v2 = v16, !v16[816]) )
  {
    v3 = v2 + 107;
    v4 = a1 & 0xFFFFFFFFC0000000LL;
    v5 = v2 + 54;
    v6 = (char *)v2 + ((a1 >> 26) & 0xF0);
    v7 = *((_QWORD *)v6 + 54);
    if ( (a1 & 0xFFFFFFFFC0000000LL) == v7 )
    {
      v8 = (__int64 *)(*((_QWORD *)v6 + 55) + ((a1 >> 9) & 0x1FFFF8));
    }
    else if ( v4 == v2[86] )
    {
      v2[86] = v7;
      v33 = v2[87];
      v2[87] = *((_QWORD *)v6 + 55);
LABEL_35:
      *((_QWORD *)v6 + 54) = v4;
      *((_QWORD *)v6 + 55) = v33;
      v8 = (__int64 *)(v33 + ((a1 >> 9) & 0x1FFFF8));
    }
    else
    {
      v37 = v2 + 88;
      for ( i = 1; i != 8; ++i )
      {
        if ( v4 == *v37 )
        {
          v39 = 2LL * i;
          v40 = &v2[2 * i - 2];
          v41 = &v2[v39];
          v33 = v41[87];
          v41[86] = v40[86];
          v41[87] = v40[87];
          v40[86] = v7;
          v40[87] = *((_QWORD *)v6 + 55);
          goto LABEL_35;
        }
        v37 += 2;
      }
      v8 = (__int64 *)sub_130D370(v2, &unk_5060AE0, v5, a1, 1, 0);
      v5 = v2 + 54;
    }
    v9 = *v8;
    v10 = HIWORD(*v8);
    v11 = qword_505FA40[(unsigned int)v10];
    if ( (v9 & 1) != 0 )
    {
      v12 = &v2[3 * (unsigned int)v10];
      v13 = &v2[3 * v10];
      v14 = v13[108];
      if ( *((_WORD *)v12 + 441) != (_WORD)v14 )
        goto LABEL_12;
      if ( !*(_WORD *)(unk_5060A20 + 2 * v10) )
        goto LABEL_53;
      v74 = qword_505FA40[(unsigned int)v10];
      sub_13108D0(
        v2,
        v2 + 107,
        &v3[3 * v10 + 1],
        (unsigned int)v10,
        (int)*(unsigned __int16 *)(unk_5060A20 + 2 * v10) >> unk_4C6F1EC);
      v34 = v13[108];
      v11 = v74;
      if ( *((_WORD *)v12 + 441) == (_WORD)v34 )
        goto LABEL_13;
    }
    else
    {
      if ( unk_5060A18 <= (unsigned int)v10 )
      {
        v30 = *((_QWORD *)v6 + 54);
        if ( v4 == v30 )
        {
          v31 = (__int64 *)(*((_QWORD *)v6 + 55) + ((a1 >> 9) & 0x1FFFF8));
        }
        else if ( v4 == v2[86] )
        {
          v2[86] = v30;
          v44 = v2[87];
          v2[87] = *((_QWORD *)v6 + 55);
LABEL_59:
          *((_QWORD *)v6 + 54) = v4;
          *((_QWORD *)v6 + 55) = v44;
          v31 = (__int64 *)(v44 + ((a1 >> 9) & 0x1FFFF8));
        }
        else
        {
          for ( j = 1; j != 8; ++j )
          {
            v54 = j;
            if ( v4 == v2[2 * j + 86] )
            {
              v55 = &v2[2 * j];
              v44 = v55[87];
              v56 = &v2[2 * (unsigned int)(v54 - 1)];
              v55[86] = v56[86];
              v55[87] = v56[87];
              v56[86] = v30;
              v56[87] = *((_QWORD *)v6 + 55);
              goto LABEL_59;
            }
          }
          v81 = v11;
          v57 = sub_130D370(v2, &unk_5060AE0, v5, a1, 1, 0);
          v11 = v81;
          v31 = (__int64 *)v57;
        }
        v73 = v11;
        v32 = *v31;
        goto LABEL_32;
      }
      v35 = &v2[3 * (unsigned int)v10];
      v13 = &v2[3 * v10];
      v14 = v13[108];
      if ( *((_WORD *)v35 + 441) != (_WORD)v14 )
      {
LABEL_12:
        v13[108] = v14 - 8;
        *(_QWORD *)(v14 - 8) = a1;
        goto LABEL_13;
      }
      v76 = qword_505FA40[(unsigned int)v10];
      sub_1310E90(
        v2,
        v2 + 107,
        &v3[3 * v10 + 1],
        (unsigned int)v10,
        (int)*(unsigned __int16 *)(unk_5060A20 + 2 * v10) >> unk_4C6F1E8);
      v34 = v13[108];
      v11 = v76;
      if ( *((_WORD *)v35 + 441) == (_WORD)v34 )
        goto LABEL_13;
    }
    v13[108] = v34 - 8;
    *(_QWORD *)(v34 - 8) = a1;
    goto LABEL_13;
  }
  if ( v16[1] )
  {
    v17 = 0;
  }
  else
  {
    v17 = v16 + 856;
    if ( !*v16 )
      v17 = 0;
  }
  v83 = a1;
  v84 = 0;
  sub_1346FC0(0, a1, &v83);
  v18 = v2 + 54;
  v19 = (char *)v2 + ((a1 >> 26) & 0xF0);
  v20 = a1 & 0xFFFFFFFFC0000000LL;
  v21 = *((_QWORD *)v19 + 54);
  if ( (a1 & 0xFFFFFFFFC0000000LL) == v21 )
  {
    v22 = (unsigned __int64 *)(*((_QWORD *)v19 + 55) + ((a1 >> 9) & 0x1FFFF8));
  }
  else if ( v20 == v2[86] )
  {
    v2[86] = v21;
    v36 = v2[87];
    v2[87] = *((_QWORD *)v19 + 55);
    *((_QWORD *)v19 + 54) = v20;
    *((_QWORD *)v19 + 55) = v36;
    v22 = (unsigned __int64 *)(v36 + ((a1 >> 9) & 0x1FFFF8));
  }
  else
  {
    for ( k = 1; k != 8; ++k )
    {
      v48 = k;
      if ( v20 == v2[2 * k + 86] )
      {
        v49 = &v2[2 * k];
        v50 = v49[87];
        v51 = &v2[2 * (unsigned int)(v48 - 1)];
        v49[86] = v51[86];
        v49[87] = v51[87];
        v51[86] = v21;
        v51[87] = *((_QWORD *)v19 + 55);
        *((_QWORD *)v19 + 55) = v50;
        *((_QWORD *)v19 + 54) = v20;
        v22 = (unsigned __int64 *)(((a1 >> 9) & 0x1FFFF8) + v50);
        goto LABEL_22;
      }
    }
    v22 = (unsigned __int64 *)sub_130D370(v2, &unk_5060AE0, v18, a1, 1, 0);
    v18 = v2 + 54;
  }
LABEL_22:
  v23 = *v22;
  v24 = v23 & 1;
  v25 = HIWORD(v23);
  v26 = (unsigned int)v25;
  v11 = qword_505FA40[(unsigned int)v25];
  if ( unk_4F969A1 )
  {
    v69 = v24;
    v68 = (unsigned int)v25;
    v70 = v25;
    v71 = v18;
    v75 = qword_505FA40[(unsigned int)v25];
    off_4C6F0B0((void *)a1, v11);
    v26 = v68;
    v24 = v69;
    v25 = v70;
    v18 = v71;
    v11 = v75;
  }
  if ( !v17 )
  {
    v77 = v11;
    sub_12FCB00((__int64)v2, a1);
    v11 = v77;
    goto LABEL_13;
  }
  if ( v24 )
  {
    v27 = &v17[24 * v26];
    v28 = &v17[24 * v25];
    v29 = *((_QWORD *)v28 + 1);
    if ( *((_WORD *)v27 + 13) != (_WORD)v29 )
    {
LABEL_27:
      *((_QWORD *)v28 + 1) = v29 - 8;
      *(_QWORD *)(v29 - 8) = a1;
      goto LABEL_13;
    }
    v43 = *(_WORD *)(unk_5060A20 + 2 * v25);
    if ( v43 )
    {
      v80 = v11;
      sub_13108D0(v2, v17, &v17[24 * v25 + 8], (unsigned int)v25, (int)v43 >> unk_4C6F1EC);
      v42 = *((_QWORD *)v28 + 1);
      v11 = v80;
      if ( *((_WORD *)v27 + 13) == (_WORD)v42 )
        goto LABEL_13;
LABEL_52:
      *((_QWORD *)v28 + 1) = v42 - 8;
      *(_QWORD *)(v42 - 8) = a1;
      goto LABEL_13;
    }
LABEL_53:
    v79 = v11;
    sub_1315B20(v2, a1);
    v11 = v79;
    goto LABEL_13;
  }
  if ( unk_5060A18 <= (unsigned int)v25 )
  {
    v45 = *((_QWORD *)v19 + 54);
    if ( v20 == v45 )
    {
      v46 = (__int64 *)(*((_QWORD *)v19 + 55) + ((a1 >> 9) & 0x1FFFF8));
    }
    else if ( v20 == v2[86] )
    {
      v2[86] = v45;
      v52 = v2[87];
      v2[87] = *((_QWORD *)v19 + 55);
LABEL_70:
      *((_QWORD *)v19 + 55) = v52;
      *((_QWORD *)v19 + 54) = v20;
      v46 = (__int64 *)(((a1 >> 9) & 0x1FFFF8) + v52);
    }
    else
    {
      for ( m = 1; m != 8; ++m )
      {
        v59 = m;
        if ( v20 == v2[2 * m + 86] )
        {
          v60 = &v2[2 * m];
          v52 = v60[87];
          v61 = &v2[2 * (unsigned int)(v59 - 1)];
          v60[86] = v61[86];
          v60[87] = v61[87];
          v61[86] = v45;
          v61[87] = *((_QWORD *)v19 + 55);
          goto LABEL_70;
        }
      }
      v82 = v11;
      v46 = (__int64 *)sub_130D370(v2, &unk_5060AE0, v18, a1, 1, 0);
      v11 = v82;
    }
    v73 = v11;
    v32 = *v46;
LABEL_32:
    sub_130A160(v2, (v32 << 16 >> 16) & 0xFFFFFFFFFFFFFF80LL);
    v11 = v73;
    goto LABEL_13;
  }
  v28 = &v17[24 * v25];
  v29 = *((_QWORD *)v28 + 1);
  if ( *(_WORD *)&v17[24 * v26 + 26] != (_WORD)v29 )
    goto LABEL_27;
  v72 = v26;
  v78 = v11;
  sub_1310E90(
    v2,
    v17,
    &v17[24 * v25 + 8],
    (unsigned int)v25,
    (int)*(unsigned __int16 *)(unk_5060A20 + 2 * v25) >> unk_4C6F1E8);
  v42 = *((_QWORD *)v28 + 1);
  v11 = v78;
  if ( *(_WORD *)&v17[24 * v72 + 26] != (_WORD)v42 )
    goto LABEL_52;
LABEL_13:
  v85 = 0;
  v86 = v2 + 105;
  v87 = v2 + 3;
  v88 = v2 + 4;
  v89 = v2 + 106;
  v15 = v2[105];
  v2[105] = v11 + v15;
  if ( v11 >= v2[4] - v15 )
    sub_13133F0(v2, &v85);
}
