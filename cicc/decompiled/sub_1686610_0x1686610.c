// Function: sub_1686610
// Address: 0x1686610
//
int __fastcall sub_1686610(
        __int64 a1,
        _BYTE *a2,
        char a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        unsigned int a7)
{
  _QWORD *v7; // rax
  __int64 *v8; // rbx
  unsigned __int64 v9; // r11
  unsigned __int64 v10; // r12
  __int64 v11; // r10
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // r8
  unsigned int v14; // r14d
  __int64 v15; // rsi
  unsigned int v16; // ecx
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // rdx
  unsigned int v19; // r15d
  __int64 v20; // rdi
  __int64 *v21; // rax
  int v22; // ecx
  int v23; // r8d
  int v24; // r9d
  __int64 v25; // rax
  _QWORD *v26; // rdi
  unsigned __int64 v27; // rax
  __int64 v28; // r8
  unsigned __int64 v29; // r15
  __int64 v30; // r9
  int v31; // esi
  int v32; // edi
  int v33; // eax
  int v34; // r14d
  _QWORD *v35; // rdi
  _QWORD *v36; // rdi
  __int64 v37; // rdx
  unsigned int v38; // ecx
  __int64 v39; // r8
  __int64 v40; // rax
  unsigned __int64 v41; // r8
  unsigned int v42; // r14d
  __int64 v43; // r9
  int v44; // esi
  int v45; // edi
  unsigned __int64 v46; // r15
  __int64 v47; // rcx
  _QWORD *v48; // rax
  __int64 v49; // rdi
  __int64 *v50; // rax
  int v51; // ecx
  int v52; // r8d
  int v53; // r9d
  __int64 v54; // rax
  __int64 v55; // rax
  unsigned int v56; // r15d
  __int64 *v57; // rax
  int v59; // [rsp+0h] [rbp-A0h]
  char v60; // [rsp+10h] [rbp-90h]
  __int64 v61; // [rsp+10h] [rbp-90h]
  int v62; // [rsp+18h] [rbp-88h]
  int v63; // [rsp+20h] [rbp-80h]
  int v65; // [rsp+30h] [rbp-70h]
  unsigned __int64 v66; // [rsp+38h] [rbp-68h]
  int v67; // [rsp+38h] [rbp-68h]
  unsigned __int64 v68; // [rsp+38h] [rbp-68h]
  unsigned __int64 v69; // [rsp+40h] [rbp-60h]
  int v70; // [rsp+40h] [rbp-60h]
  int v71; // [rsp+40h] [rbp-60h]
  int v72; // [rsp+40h] [rbp-60h]
  unsigned __int64 v73; // [rsp+40h] [rbp-60h]
  __int64 v74; // [rsp+40h] [rbp-60h]
  int v75; // [rsp+48h] [rbp-58h]
  unsigned int v76; // [rsp+48h] [rbp-58h]
  __int64 v77; // [rsp+48h] [rbp-58h]
  unsigned int v78; // [rsp+48h] [rbp-58h]
  __int64 v79; // [rsp+48h] [rbp-58h]
  unsigned __int64 v81; // [rsp+48h] [rbp-58h]
  unsigned __int64 v82; // [rsp+48h] [rbp-58h]
  _BYTE *v83; // [rsp+50h] [rbp-50h]
  __int64 v85; // [rsp+60h] [rbp-40h]
  unsigned __int64 v86; // [rsp+68h] [rbp-38h]
  unsigned __int64 v87; // [rsp+68h] [rbp-38h]
  __int64 v88; // [rsp+68h] [rbp-38h]
  unsigned __int64 v89; // [rsp+68h] [rbp-38h]
  unsigned __int64 v91; // [rsp+68h] [rbp-38h]

  LODWORD(v7) = a7;
  v85 = a1;
  v83 = a2;
  if ( !a5 )
    return (int)v7;
  v8 = *(__int64 **)a1;
  v9 = a4;
  v10 = a5;
  v11 = a6;
  v12 = a4 + a5 - 1;
  if ( !*(_QWORD *)a1 )
  {
    if ( a7 )
    {
      a2 = (_BYTE *)(a5 - 1);
      a1 = -1;
      v38 = a7;
      while ( 1 )
      {
        if ( v38 <= 0x3F )
        {
          v39 = ~(-1LL << v38);
          v40 = v9 & v39;
          v41 = v12 & v39;
        }
        else
        {
          v41 = v12;
          v40 = v9;
        }
        if ( (_BYTE *)(v41 - v40) != a2 )
          break;
        v38 -= 4;
        if ( !v38 )
          goto LABEL_64;
      }
      v14 = v38;
      v81 = v9;
      v56 = v38 + 4;
      v49 = *(_QWORD *)(sub_1689050(-1, a2, v12) + 24);
      v57 = sub_1685080(v49, 160);
      v51 = v14;
      v11 = a6;
      v9 = v81;
      v17 = 1LL << v14;
      v18 = (1LL << v14) - 1;
      if ( v57 )
      {
        v8 = v57;
LABEL_70:
        v8[1] = 0;
        v8[19] = 0;
        v55 = 0;
        memset(
          (void *)((unsigned __int64)(v8 + 2) & 0xFFFFFFFFFFFFFFF8LL),
          0,
          8LL * (((unsigned int)v8 - (((_DWORD)v8 + 16) & 0xFFFFFFF8) + 160) >> 3));
        if ( v56 > 0x3F )
          goto LABEL_67;
        v54 = ~(-1LL << v56);
LABEL_66:
        v55 = v9 & ~v54;
LABEL_67:
        *v8 = v55;
        *((_DWORD *)v8 + 2) = v14;
        *(_QWORD *)v85 = v8;
        goto LABEL_20;
      }
    }
    else
    {
LABEL_64:
      v89 = v9;
      v49 = *(_QWORD *)(sub_1689050(a1, a2, v12) + 24);
      v50 = sub_1685080(v49, 160);
      v9 = v89;
      v11 = a6;
      if ( v50 )
      {
        v8 = v50;
        v17 = 1;
        v50[1] = 0;
        v14 = 0;
        v50[19] = 0;
        v18 = 0;
        memset(
          (void *)((unsigned __int64)(v50 + 2) & 0xFFFFFFFFFFFFFFF8LL),
          0,
          8LL * (((unsigned int)v50 - (((_DWORD)v50 + 16) & 0xFFFFFFF8) + 160) >> 3));
        v54 = 15;
        goto LABEL_66;
      }
      v18 = 0;
      v17 = 1;
      v56 = 4;
      v14 = 0;
    }
    v74 = v11;
    v82 = v9;
    v91 = v18;
    sub_1683C30(v49, 160, v18, v51, v52, v53, v60);
    v18 = v91;
    v9 = v82;
    v11 = v74;
    goto LABEL_70;
  }
  v13 = *v8;
  v14 = *((_DWORD *)v8 + 2);
  v15 = a4;
  if ( *v8 <= a4 )
    v15 = *v8;
  if ( v13 + (16LL << v14) - 1 >= v12 )
    v12 = v13 + (16LL << v14) - 1;
  v16 = a7;
  if ( !a7 )
  {
LABEL_56:
    if ( !v14 )
    {
      v27 = v10;
      LOBYTE(v46) = v9;
      v18 = 0;
      v28 = 0;
      v17 = 1;
      goto LABEL_59;
    }
    v75 = -4;
    v18 = 0;
    v17 = 1;
    v14 = 0;
    v19 = 4;
    goto LABEL_14;
  }
  while ( v16 > 0x3F || v12 - v15 == (~(-1LL << v16) & v12) - (v15 & ~(-1LL << v16)) )
  {
    v16 -= 4;
    if ( !v16 )
      goto LABEL_56;
  }
  v17 = 1LL << v16;
  v18 = (1LL << v16) - 1;
  if ( v14 != v16 )
  {
    v19 = v16 + 4;
    v13 >>= v16;
    v14 = v16;
    v75 = v16 - 4;
LABEL_14:
    v66 = v9;
    v69 = v18;
    v86 = v13 & 0xF;
    v20 = *(_QWORD *)(sub_1689050(-1, v15, v18) + 24);
    v21 = sub_1685080(v20, 160);
    v18 = v69;
    v9 = v66;
    v11 = a6;
    v8 = v21;
    if ( !v21 )
    {
      sub_1683C30(v20, 160, v69, v22, v23, v24, v60);
      v11 = a6;
      v9 = v66;
      v18 = v69;
    }
    v8[1] = 0;
    v8[19] = 0;
    v25 = 0;
    memset(
      (void *)((unsigned __int64)(v8 + 2) & 0xFFFFFFFFFFFFFFF8LL),
      0,
      8LL * (((unsigned int)v8 - (((_DWORD)v8 + 16) & 0xFFFFFFF8) + 160) >> 3));
    if ( v19 <= 0x3F )
      v25 = v9 & (-1LL << v19);
    *v8 = v25;
    *((_DWORD *)v8 + 2) = v14;
    v26 = *(_QWORD **)v85;
    v8[v86 + 4] = *(_QWORD *)v85;
    if ( *((_DWORD *)v26 + 2) == v75 )
    {
      v47 = v26[4];
      v48 = v26 + 5;
      while ( v47 == *v48 )
      {
        if ( v26 + 20 == ++v48 )
        {
          v8[v86 + 4] = v47;
          v68 = v18;
          *((_BYTE *)v8 + v86 + 12) = 1;
          v73 = v9;
          v79 = v11;
          sub_16856A0(v26);
          v11 = v79;
          v9 = v73;
          v18 = v68;
          break;
        }
      }
    }
    *(_QWORD *)v85 = v8;
LABEL_20:
    if ( v14 > 0x3F )
    {
      v27 = v10 + v9;
      v28 = v9;
      v29 = 0;
      goto LABEL_22;
    }
  }
  v46 = v9 >> v14;
  v28 = v9 & ~(-1LL << v14);
  v27 = v10 + v28;
LABEL_59:
  v29 = v46 & 0xF;
LABEL_22:
  if ( v27 - 1 >= v18 )
  {
    v70 = 0;
    v87 = v17 - v28;
  }
  else
  {
    v87 = v10;
    v70 = v17 - v27;
  }
  if ( v28 )
  {
    v76 = v14 - 4;
    v30 = v8[v29 + 4];
    v31 = (_DWORD)v8 + v29 + 12;
    v32 = (_DWORD)v8 + 8 * (v29 + 4);
    v33 = v87 + v9;
    if ( *((_BYTE *)v8 + v29 + 12) && v30 )
    {
      v62 = v87 + v9;
      v8[v29 + 4] = 0;
      *((_BYTE *)v8 + v29 + 12) = 0;
      v61 = v11;
      v63 = v9;
      v65 = v30;
      sub_1686610(v32, v31, 0, v9 - v28, v28, v30, v76);
      sub_1686610((_DWORD)v8 + 8 * (v29 + 4), (_DWORD)v8 + v29 + 12, 0, v62, v70, v65, v76);
      LODWORD(v9) = v63;
      v33 = v62;
      v11 = v61;
      v31 = (_DWORD)v8 + v29 + 12;
      v32 = (_DWORD)v8 + 8 * (v29 + 4);
    }
    v71 = v33;
    v77 = v11;
    ++v29;
    sub_1686610(v32, v31, 0, v9, v87, v11, v14 - 4);
    v10 -= v87;
    v11 = v77;
    LODWORD(v7) = v71;
    if ( v10 < v17 )
    {
LABEL_35:
      if ( !v10 )
        goto LABEL_36;
      goto LABEL_51;
    }
LABEL_29:
    v78 = v14;
    v34 = (int)v7;
    do
    {
      if ( !*((_BYTE *)v8 + v29 + 12) )
      {
        v35 = (_QWORD *)v8[v29 + 4];
        if ( v35 )
        {
          v88 = v11;
          sub_1686480(v35);
          v11 = v88;
        }
      }
      v10 -= v17;
      v8[v29 + 4] = v11;
      v34 += v17;
      *((_BYTE *)v8 + v29++ + 12) = 1;
    }
    while ( v10 >= v17 );
    LODWORD(v7) = v34;
    v14 = v78;
    goto LABEL_35;
  }
  LODWORD(v7) = v9;
  if ( v10 >= v17 )
    goto LABEL_29;
LABEL_51:
  v42 = v14 - 4;
  v43 = v8[v29 + 4];
  v44 = (_DWORD)v8 + v29 + 12;
  v45 = (_DWORD)v8 + 8 * (v29 + 4);
  if ( *((_BYTE *)v8 + v29 + 12) && v43 )
  {
    v67 = v11;
    v8[v29 + 4] = 0;
    *((_BYTE *)v8 + v29 + 12) = 0;
    v72 = (int)v7;
    sub_1686610(v45, v44, 0, v10 + (_DWORD)v7, v17 - v10, v43, v42);
    LODWORD(v11) = v67;
    LODWORD(v7) = v72;
    v44 = (_DWORD)v8 + v29 + 12;
    v45 = (_DWORD)v8 + 8 * (v29 + 4);
  }
  sub_1686610(v45, v44, 0, (_DWORD)v7, v10, v11, v42);
  LODWORD(v7) = v59;
LABEL_36:
  if ( !a3 )
  {
    v36 = *(_QWORD **)v85;
    LODWORD(v7) = a7;
    if ( a7 == *(_DWORD *)(*(_QWORD *)v85 + 8LL) )
    {
      v37 = v36[4];
      v7 = v36 + 5;
      while ( v37 == *v7 )
      {
        if ( v36 + 20 == ++v7 )
        {
          *(_QWORD *)v85 = v37;
          *v83 = 1;
          LODWORD(v7) = sub_16856A0(v36);
          return (int)v7;
        }
      }
    }
  }
  return (int)v7;
}
