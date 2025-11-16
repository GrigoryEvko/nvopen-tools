// Function: sub_2D0C690
// Address: 0x2d0c690
//
__int64 __fastcall sub_2D0C690(__int64 a1)
{
  _QWORD *v1; // r14
  _BYTE *v3; // rsi
  _QWORD *v4; // rax
  _QWORD *v5; // rax
  unsigned __int64 v6; // rdi
  _QWORD *v7; // r8
  __int64 v8; // r12
  _QWORD *v9; // r15
  bool v10; // zf
  unsigned __int16 v11; // ax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // r12
  __int64 v23; // rsi
  __int64 *v24; // rbx
  __int64 *v25; // rbx
  __int64 *v26; // r12
  __int64 v27; // rdi
  __int64 v28; // rax
  unsigned __int16 v29; // ax
  __int64 *v31; // r8
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rcx
  unsigned int **v35; // rbx
  unsigned int **v36; // rcx
  unsigned int *v37; // rax
  int v38; // edx
  __int64 v39; // rcx
  unsigned int *v40; // rax
  unsigned int **v41; // r10
  int v42; // edx
  int v43; // r11d
  unsigned int *v44; // rax
  int v45; // edx
  int v46; // r11d
  unsigned int *v47; // rax
  int v48; // edx
  int v49; // r11d
  __int64 *v50; // r15
  __int64 v51; // rdx
  char v52; // al
  char v53; // di
  char *v54; // rax
  __int64 *v55; // rax
  char v56; // al
  char v57; // al
  char v58; // al
  __int64 *v59; // [rsp+0h] [rbp-100h]
  __int64 *v60; // [rsp+0h] [rbp-100h]
  __int64 v61; // [rsp+8h] [rbp-F8h]
  __int64 *v62; // [rsp+8h] [rbp-F8h]
  __int64 *v63; // [rsp+10h] [rbp-F0h]
  unsigned __int8 v64; // [rsp+1Eh] [rbp-E2h]
  char v65; // [rsp+1Fh] [rbp-E1h]
  __int64 v66; // [rsp+28h] [rbp-D8h] BYREF
  _QWORD *v67; // [rsp+30h] [rbp-D0h] BYREF
  _QWORD *v68; // [rsp+38h] [rbp-C8h]
  _BYTE *v69; // [rsp+40h] [rbp-C0h]
  __int64 v70; // [rsp+50h] [rbp-B0h] BYREF
  char *v71; // [rsp+58h] [rbp-A8h]
  __int64 v72; // [rsp+60h] [rbp-A0h]
  int v73; // [rsp+68h] [rbp-98h]
  char v74; // [rsp+6Ch] [rbp-94h]
  char v75; // [rsp+70h] [rbp-90h] BYREF
  _QWORD *v76; // [rsp+90h] [rbp-70h] BYREF
  char *v77; // [rsp+98h] [rbp-68h]
  __int64 v78; // [rsp+A0h] [rbp-60h]
  int v79; // [rsp+A8h] [rbp-58h]
  char v80; // [rsp+ACh] [rbp-54h]
  char v81; // [rsp+B0h] [rbp-50h] BYREF

  v1 = *(_QWORD **)(a1 + 104);
  v67 = 0;
  v68 = 0;
  v69 = 0;
  if ( v1 == (_QWORD *)(a1 + 104) )
    return 0;
  v3 = 0;
  v4 = 0;
  while ( 1 )
  {
    v76 = v1 + 2;
    if ( v3 == (_BYTE *)v4 )
    {
      sub_2D07C30((__int64)&v67, v3, &v76);
      v5 = v68;
    }
    else
    {
      if ( v4 )
      {
        *v4 = v1 + 2;
        v4 = v68;
      }
      v5 = v4 + 1;
      v68 = v5;
    }
    sub_2D04420((__int64)v67, v5 - v67 - 1, 0, *(v5 - 1));
    v1 = (_QWORD *)*v1;
    if ( v1 == (_QWORD *)(a1 + 104) )
      break;
    v4 = v68;
    v3 = v69;
  }
  v6 = (unsigned __int64)v68;
  v7 = v67;
  if ( v67 == v68 )
  {
    v64 = 0;
    goto LABEL_37;
  }
  v64 = 0;
  v8 = *v67;
  v9 = &v76;
  if ( (char *)v68 - (char *)v67 > 8 )
    goto LABEL_53;
  while ( 1 )
  {
    v10 = *(_BYTE *)(v8 + 41) == 0;
    v6 = (unsigned __int64)--v68;
    if ( !v10 )
      goto LABEL_51;
    v11 = sub_2D06360(*(_QWORD *)(a1 + 48), 1.0);
    if ( !(_BYTE)v11 || *(int *)(v8 + 8) <= 0 )
    {
      if ( !HIBYTE(v11) )
        goto LABEL_50;
      if ( *(int *)(v8 + 12) <= 0 )
      {
        v6 = (unsigned __int64)v68;
        goto LABEL_51;
      }
    }
    v80 = 1;
    v71 = &v75;
    v70 = 0;
    v72 = 4;
    v73 = 0;
    v74 = 1;
    v76 = 0;
    v77 = &v81;
    v78 = 4;
    v79 = 0;
    v65 = sub_2D04210(*(_QWORD *)v8);
    if ( v65 )
    {
      v15 = *(_QWORD *)v8;
      if ( **(_BYTE **)v8 != 22 )
        v15 = 0;
      if ( !(unsigned __int8)sub_2D053E0(a1, v15, &v70, v12, v13, v14) )
        goto LABEL_26;
      goto LABEL_19;
    }
    sub_2D0A1E0(a1, v8);
    v15 = v8;
    sub_2D0C670((__int64 *)a1, (_BYTE **)v8);
    v31 = *(__int64 **)(v8 + 48);
    v32 = 8LL * *(unsigned int *)(v8 + 56);
    v63 = &v31[(unsigned __int64)v32 / 8];
    v33 = v32 >> 3;
    v34 = v32 >> 5;
    if ( v34 )
    {
      v35 = *(unsigned int ***)(v8 + 48);
      v36 = (unsigned int **)&v31[4 * v34];
      while ( 1 )
      {
        v37 = *v35;
        v38 = (*v35)[4];
        if ( v38 < 0 )
          goto LABEL_46;
        v15 = v37[5];
        if ( (int)v15 < 0 )
          goto LABEL_46;
        if ( !((unsigned int)v15 | v38) )
          goto LABEL_46;
        v15 = *(unsigned int *)(a1 + 64);
        if ( (int)v37[6] > (int)v15 || v37[7] > dword_50158A8 || v37[8] > dword_50157C8 )
          goto LABEL_46;
        v40 = v35[1];
        v41 = v35 + 1;
        v42 = v40[4];
        if ( v42 < 0 )
          goto LABEL_91;
        v43 = v40[5];
        if ( v43 < 0 )
          goto LABEL_91;
        if ( !(v43 | v42) )
          goto LABEL_91;
        if ( (int)v15 < (int)v40[6] )
          goto LABEL_91;
        if ( dword_50158A8 < v40[7] )
          goto LABEL_91;
        if ( dword_50157C8 < v40[8] )
          goto LABEL_91;
        v44 = v35[2];
        v41 = v35 + 2;
        v45 = v44[4];
        if ( v45 < 0
          || (v46 = v44[5], v46 < 0)
          || !(v46 | v45)
          || (int)v15 < (int)v44[6]
          || dword_50158A8 < v44[7]
          || dword_50157C8 < v44[8]
          || (v47 = v35[3], v41 = v35 + 3, v48 = v47[4], v48 < 0)
          || (v49 = v47[5], v49 < 0)
          || !(v49 | v48)
          || (int)v15 < (int)v47[6]
          || dword_50158A8 < v47[7]
          || dword_50157C8 < v47[8] )
        {
LABEL_91:
          v35 = v41;
          goto LABEL_46;
        }
        v35 += 4;
        if ( v35 == v36 )
        {
          v33 = ((char *)v63 - (char *)v35) >> 3;
          goto LABEL_80;
        }
      }
    }
    v35 = *(unsigned int ***)(v8 + 48);
LABEL_80:
    if ( v33 != 2 )
      break;
    v15 = a1 + 56;
LABEL_117:
    v60 = v31;
    v58 = sub_2D04F60((int *)*v35 + 4, v15);
    v31 = v60;
    if ( v58 )
    {
      ++v35;
      goto LABEL_112;
    }
LABEL_46:
    if ( v63 == (__int64 *)v35 )
      goto LABEL_83;
LABEL_47:
    if ( !v80 )
      _libc_free((unsigned __int64)v77);
    if ( v74 )
    {
LABEL_50:
      v6 = (unsigned __int64)v68;
      goto LABEL_51;
    }
    _libc_free((unsigned __int64)v71);
    v6 = (unsigned __int64)v68;
LABEL_51:
    v7 = v67;
    if ( v67 == (_QWORD *)v6 )
      goto LABEL_37;
    v8 = *v67;
    if ( (__int64)(v6 - (_QWORD)v67) > 8 )
    {
LABEL_53:
      v39 = *(_QWORD *)(v6 - 8);
      *(_QWORD *)(v6 - 8) = v8;
      sub_2D04510((__int64)v7, 0, (__int64)(v6 - 8 - (_QWORD)v7) >> 3, v39);
    }
  }
  if ( v33 == 3 )
  {
    v15 = a1 + 56;
    v59 = *(__int64 **)(v8 + 48);
    v57 = sub_2D04F60((int *)*v35 + 4, a1 + 56);
    v31 = v59;
    if ( !v57 )
      goto LABEL_46;
    v15 = a1 + 56;
    ++v35;
    goto LABEL_117;
  }
  if ( v33 != 1 )
  {
LABEL_83:
    if ( v31 == v63 )
      goto LABEL_26;
    goto LABEL_84;
  }
  v15 = a1 + 56;
LABEL_112:
  v62 = v31;
  v56 = sub_2D04F60((int *)*v35 + 4, v15);
  v31 = v62;
  if ( !v56 )
    goto LABEL_46;
  if ( v62 == v63 )
    goto LABEL_26;
LABEL_84:
  v61 = (__int64)v9;
  v50 = v31;
  do
  {
    v51 = *v50;
    v15 = v8;
    v66 = 0;
    v52 = sub_2D0A020(a1, (__int64 *)v8, v51, &v66, &v70);
    if ( v52 )
    {
      v15 = v66;
      v65 = v52;
      if ( v66 )
        sub_AE6EC0(v61, v66);
    }
    ++v50;
  }
  while ( v63 != v50 );
  v9 = (_QWORD *)v61;
  if ( !v65 )
    goto LABEL_26;
LABEL_19:
  v64 = qword_5015368;
  if ( (_BYTE)qword_5015368 )
  {
    v20 = *(__int64 **)(v8 + 136);
    if ( *(_BYTE *)(v8 + 156) )
      v21 = *(unsigned int *)(v8 + 148);
    else
      v21 = *(unsigned int *)(v8 + 144);
    v22 = &v20[v21];
    if ( v20 != v22 )
    {
      while ( 1 )
      {
        v23 = *v20;
        v24 = v20;
        if ( (unsigned __int64)*v20 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v22 == ++v20 )
          goto LABEL_25;
      }
      if ( v22 != v20 )
      {
        v53 = v80;
        if ( v80 )
        {
LABEL_95:
          v54 = v77;
          v17 = HIDWORD(v78);
          v21 = (__int64)&v77[8 * HIDWORD(v78)];
          if ( v77 != (char *)v21 )
          {
            do
            {
              if ( *(_QWORD *)v54 == v23 )
                goto LABEL_99;
              v54 += 8;
            }
            while ( (char *)v21 != v54 );
          }
          if ( HIDWORD(v78) < (unsigned int)v78 )
          {
            v17 = (unsigned int)++HIDWORD(v78);
            *(_QWORD *)v21 = v23;
            v53 = v80;
            v76 = (_QWORD *)((char *)v76 + 1);
            goto LABEL_99;
          }
        }
        while ( 1 )
        {
          sub_C8CC70((__int64)v9, v23, v21, v17, v18, v19);
          v53 = v80;
LABEL_99:
          v55 = v24 + 1;
          if ( v24 + 1 == v22 )
            break;
          v23 = *v55;
          for ( ++v24; (unsigned __int64)*v55 >= 0xFFFFFFFFFFFFFFFELL; v24 = v55 )
          {
            if ( v22 == ++v55 )
              goto LABEL_25;
            v23 = *v55;
          }
          if ( v22 == v24 )
            break;
          if ( v53 )
            goto LABEL_95;
        }
      }
    }
LABEL_25:
    v15 = (__int64)&v70;
    sub_FD21F0(*(_QWORD *)(a1 + 48), (__int64)&v70, (__int64)v9);
    if ( (_BYTE)qword_5015288 )
    {
      v15 = (unsigned __int8)qword_50151A8;
      if ( !(unsigned __int8)sub_2D0B540(*(_QWORD *)(a1 + 48), qword_50151A8) )
        sub_C64ED0("Incorrect RP info from incremental RPA update in Rematerialization.\n", 1u);
    }
  }
  else
  {
    sub_2D0B0C0(*(_QWORD *)(a1 + 48), v15, v16, v17);
    v64 = 1;
  }
LABEL_26:
  v25 = *(__int64 **)(a1 + 280);
  v26 = *(__int64 **)(a1 + 288);
  if ( v26 != v25 )
  {
    do
    {
      v27 = *v25++;
      sub_BD72D0(v27, v15);
    }
    while ( v26 != v25 );
    v28 = *(_QWORD *)(a1 + 280);
    if ( v28 != *(_QWORD *)(a1 + 288) )
      *(_QWORD *)(a1 + 288) = v28;
  }
  v29 = sub_2D06360(*(_QWORD *)(a1 + 48), 1.0);
  if ( (_BYTE)v29 || HIBYTE(v29) )
    goto LABEL_47;
  if ( !v80 )
    _libc_free((unsigned __int64)v77);
  if ( !v74 )
    _libc_free((unsigned __int64)v71);
  v6 = (unsigned __int64)v67;
LABEL_37:
  if ( v6 )
    j_j___libc_free_0(v6);
  return v64;
}
