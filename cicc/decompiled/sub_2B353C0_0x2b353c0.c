// Function: sub_2B353C0
// Address: 0x2b353c0
//
__int64 __fastcall sub_2B353C0(_QWORD *a1, __int64 a2, char a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r9
  char v6; // r14
  _QWORD *v7; // r12
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // r10
  int v12; // r13d
  unsigned __int64 v13; // rax
  int *v14; // rcx
  __int64 v15; // rdx
  _BYTE *v16; // rdx
  _BYTE *v17; // r13
  unsigned int v18; // r12d
  int *v19; // rdi
  __int64 v21; // rsi
  unsigned int v22; // edx
  char v23; // al
  char v24; // al
  __int64 v25; // r14
  __int64 v26; // r11
  __int64 v27; // r8
  int *v28; // r10
  int v29; // r13d
  char v30; // al
  __int64 v31; // rcx
  unsigned int v32; // edx
  __int64 v33; // rax
  unsigned __int64 v34; // rsi
  unsigned int v35; // r13d
  unsigned __int64 v36; // rax
  int *v37; // rdi
  __int64 v38; // rcx
  unsigned int v39; // edx
  __int64 v40; // rcx
  _BYTE *v41; // rsi
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rsi
  int *v48; // rsi
  __int64 v49; // rax
  char v50; // al
  __int64 v51; // rax
  int *v52; // rsi
  int *v53; // rcx
  __int64 v54; // rsi
  unsigned __int64 v55; // rax
  signed int v56; // [rsp+0h] [rbp-140h]
  char v57; // [rsp+0h] [rbp-140h]
  unsigned int v58; // [rsp+0h] [rbp-140h]
  __int64 v59; // [rsp+8h] [rbp-138h]
  int *v60; // [rsp+8h] [rbp-138h]
  __int64 v61; // [rsp+10h] [rbp-130h]
  char v62; // [rsp+10h] [rbp-130h]
  __int64 v63; // [rsp+10h] [rbp-130h]
  signed int v64; // [rsp+1Ch] [rbp-124h]
  unsigned int v65; // [rsp+1Ch] [rbp-124h]
  __int64 v67; // [rsp+20h] [rbp-120h]
  int *v68; // [rsp+20h] [rbp-120h]
  __int64 v71; // [rsp+28h] [rbp-118h]
  __int64 v72; // [rsp+28h] [rbp-118h]
  __int64 v73; // [rsp+30h] [rbp-110h]
  unsigned __int64 *v74; // [rsp+48h] [rbp-F8h] BYREF
  int *v75; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v76; // [rsp+58h] [rbp-E8h]
  _BYTE v77[48]; // [rsp+60h] [rbp-E0h] BYREF
  _BYTE *v78; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v79; // [rsp+98h] [rbp-A8h]
  _BYTE v80[48]; // [rsp+A0h] [rbp-A0h] BYREF
  unsigned __int64 *v81[2]; // [rsp+D0h] [rbp-70h] BYREF
  _BYTE v82[96]; // [rsp+E0h] [rbp-60h] BYREF

  v5 = 0xC00000000LL;
  v6 = a3;
  v7 = a1;
  v9 = *a1;
  v75 = (int *)v77;
  v76 = 0xC00000000LL;
  if ( *(_BYTE *)v9 != 92 )
  {
    v10 = *(_QWORD *)(v9 + 8);
    if ( *(_BYTE *)(v10 + 8) != 17 )
    {
      v19 = (int *)v77;
      goto LABEL_59;
    }
    v73 = 0;
    goto LABEL_4;
  }
  v73 = 0;
  while ( 1 )
  {
    v25 = *(_QWORD *)(v9 + 8);
    if ( *(_BYTE *)(v25 + 8) != 17 )
    {
LABEL_55:
      v7 = a1;
      v6 = a3;
      goto LABEL_56;
    }
    v26 = *(unsigned int *)(a2 + 8);
    LODWORD(v27) = *(_DWORD *)(v25 + 32);
    LODWORD(v81[0]) = -1;
    v28 = *(int **)a2;
    v29 = v26;
    if ( (_DWORD)v26 == (_DWORD)v27 )
    {
      v58 = v27;
      v60 = *(int **)a2;
      v63 = v26;
      v50 = sub_B4ED80(*(int **)a2, v26, v26);
      v26 = v63;
      v28 = v60;
      v27 = v58;
      if ( v50 )
        goto LABEL_87;
    }
    if ( (v56 = v27,
          v59 = (__int64)v28,
          v61 = v26,
          v30 = sub_B4EFF0(v28, v26, v27, (int *)v81),
          v27 = (unsigned int)v56,
          v30)
      && !LODWORD(v81[0])
      || !(v29 % v56) && sub_2B1ACD0(0, v29 / v56, v59, v61, v56) )
    {
LABEL_87:
      if ( !v73
        || a3 != 1
        || (v47 = *(unsigned int *)(a2 + 8), (_DWORD)v47 == *(_DWORD *)(v25 + 32))
        && (unsigned __int8)sub_B4ED80(*(int **)a2, v47, v47)
        && !(unsigned __int8)sub_B4EE20(v75, (unsigned int)v76, v76) )
      {
        v48 = *(int **)a2;
        v49 = *(unsigned int *)(a2 + 8);
        LODWORD(v76) = 0;
        sub_2B35330((__int64)&v75, v48, &v48[v49], v31, v27, 0xC00000000LL);
        v73 = v9;
      }
    }
    v32 = *(_DWORD *)(v9 + 80);
    v33 = *(_QWORD *)(*(_QWORD *)(v9 - 64) + 8LL);
    if ( *(_DWORD *)(v33 + 32) == v32 )
    {
      if ( (unsigned __int8)sub_B4EE20(*(int **)(v9 + 72), v32, v32) )
      {
        v51 = *(unsigned int *)(a2 + 8);
        v52 = *(int **)a2;
        LODWORD(v76) = 0;
        sub_2B35330((__int64)&v75, v52, &v52[v51], v31, v27, 0xC00000000LL);
        v73 = v9;
      }
      v33 = *(_QWORD *)(*(_QWORD *)(v9 - 64) + 8LL);
    }
    v34 = *(unsigned int *)(a2 + 8);
    v35 = *(_DWORD *)(a2 + 8);
    if ( *(_BYTE *)(v33 + 8) == 17 )
      v35 = *(_DWORD *)(v33 + 32);
    v79 = 0xC00000000LL;
    v78 = v80;
    sub_11B1960((__int64)&v78, v34, -1, v31, v27, 0xC00000000LL);
    v36 = 0;
    v37 = *(int **)a2;
    v38 = 4LL * *(unsigned int *)(a2 + 8);
    if ( v38 )
    {
      do
      {
        v39 = v37[v36 / 4];
        if ( v39 != -1 && *(_DWORD *)(v9 + 80) > v39 )
          *(_DWORD *)&v78[v36] = *(_DWORD *)(*(_QWORD *)(v9 + 72) + 4LL * v39);
        v36 += 4LL;
      }
      while ( v38 != v36 );
    }
    sub_2B23C00((__int64 *)&v74, v35, (__int64)v78, (unsigned int)v79, 0);
    sub_2B25530(v81, *(_QWORD *)(v9 - 64), (unsigned __int64 *)&v74);
    v57 = sub_2B0D9E0((unsigned __int64)v81[0]);
    sub_228BF40(v81);
    sub_228BF40(&v74);
    sub_2B23C00((__int64 *)&v74, v35, (__int64)v78, (unsigned int)v79, 1);
    sub_2B25530(v81, *(_QWORD *)(v9 - 32), (unsigned __int64 *)&v74);
    v62 = sub_2B0D9E0((unsigned __int64)v81[0]);
    sub_228BF40(v81);
    sub_228BF40(&v74);
    v5 = 0xC00000000LL;
    if ( !v57 && !v62 )
      break;
    v41 = *(_BYTE **)(v9 + 72);
    v42 = *(unsigned int *)(v9 + 80);
    v81[0] = (unsigned __int64 *)v82;
    v81[1] = (unsigned __int64 *)0xC00000000LL;
    sub_2B35330((__int64)v81, v41, &v41[4 * v42], v40, a5, 0xC00000000LL);
    sub_2B31420(v35, (__int64)v81, *(_QWORD *)a2, *(unsigned int *)(a2 + 8));
    sub_2B310D0(a2, (__int64)v81, v43, v44, v45, v46);
    if ( v62 )
      v9 = *(_QWORD *)(v9 - 64);
    else
      v9 = *(_QWORD *)(v9 - 32);
    if ( (_BYTE *)v81[0] != v82 )
      _libc_free((unsigned __int64)v81[0]);
    if ( v78 != v80 )
      _libc_free((unsigned __int64)v78);
    if ( *(_BYTE *)v9 != 92 )
      goto LABEL_55;
  }
  v53 = *(int **)a2;
  v7 = a1;
  v6 = a3;
  v54 = *(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8);
  if ( v54 != *(_QWORD *)a2 )
  {
    do
    {
      v55 = *v53;
      if ( (_DWORD)v55 != -1 && *(_DWORD *)(*(_QWORD *)(v9 + 72) + 4 * (v55 % *(unsigned int *)(v9 + 80))) == -1 )
        *v53 = -1;
      ++v53;
    }
    while ( (int *)v54 != v53 );
  }
  if ( v78 != v80 )
    _libc_free((unsigned __int64)v78);
LABEL_56:
  v10 = *(_QWORD *)(v9 + 8);
  if ( *(_BYTE *)(v10 + 8) == 17 )
  {
LABEL_4:
    v11 = *(unsigned int *)(a2 + 8);
    a5 = *(unsigned int *)(v10 + 32);
    LODWORD(v81[0]) = -1;
    v5 = *(_QWORD *)a2;
    v12 = v11;
    if ( (_DWORD)v11 == (_DWORD)a5 )
    {
      v65 = a5;
      v68 = *(int **)a2;
      v72 = v11;
      v24 = sub_B4ED80(*(int **)a2, v11, v11);
      v11 = v72;
      v5 = (__int64)v68;
      a5 = v65;
      if ( v24 )
        goto LABEL_88;
    }
    if ( !v6 )
    {
      if ( (v64 = a5,
            v67 = v5,
            v71 = v11,
            v23 = sub_B4EFF0((int *)v5, v11, a5, (int *)v81),
            v5 = v67,
            a5 = (unsigned int)v64,
            v23)
        && !LODWORD(v81[0])
        || !(v12 % v64) && sub_2B1ACD0(0, v12 / v64, v67, v71, v64) )
      {
LABEL_88:
        if ( !(unsigned __int8)sub_B4EE20(*(int **)a2, *(unsigned int *)(a2 + 8), *(_DWORD *)(a2 + 8)) )
        {
          *v7 = v9;
          v19 = v75;
          v18 = 1;
          goto LABEL_16;
        }
      }
    }
  }
  if ( v73 )
  {
    *v7 = v73;
    v13 = 0;
    v14 = *(int **)a2;
    v15 = 4LL * *(unsigned int *)(a2 + 8);
    if ( v15 )
    {
      do
      {
        if ( v14[v13 / 4] == -1 )
          v75[v13 / 4] = -1;
        v13 += 4LL;
      }
      while ( v15 != v13 );
    }
    sub_2B310D0(a2, (__int64)&v75, v15, (__int64)v14, a5, v5);
    v16 = 0;
    if ( *(_BYTE *)*v7 == 92 )
      v16 = (_BYTE *)*v7;
    v17 = v16;
    if ( !v6 )
      goto LABEL_14;
    v21 = *(unsigned int *)(a2 + 8);
    if ( (_DWORD)v21 == *(_DWORD *)(*(_QWORD *)(*v7 + 8LL) + 32LL) )
    {
      v18 = sub_B4ED80(*(int **)a2, v21, v21);
      if ( (_BYTE)v18 )
        goto LABEL_15;
    }
    if ( v17
      && (v22 = *((_DWORD *)v17 + 20), v22 == *(_DWORD *)(a2 + 8))
      && v22 == *(_DWORD *)(*(_QWORD *)(*((_QWORD *)v17 - 8) + 8LL) + 32LL)
      && (unsigned __int8)sub_B4EE20(*((int **)v17 + 9), v22, v22) )
    {
      v18 = sub_B4EE20(*(int **)a2, *(unsigned int *)(a2 + 8), *(_DWORD *)(a2 + 8));
    }
    else
    {
LABEL_14:
      v18 = 0;
    }
LABEL_15:
    v19 = v75;
    goto LABEL_16;
  }
  v19 = v75;
LABEL_59:
  *v7 = v9;
  v18 = 0;
LABEL_16:
  if ( v19 != (int *)v77 )
    _libc_free((unsigned __int64)v19);
  return v18;
}
