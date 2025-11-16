// Function: sub_31E3430
// Address: 0x31e3430
//
void __fastcall sub_31E3430(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  bool v5; // r15
  unsigned int v6; // esi
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // r13
  int v10; // ecx
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // r14
  unsigned int v14; // eax
  __int64 v15; // r13
  __int64 v16; // rdi
  unsigned int v17; // ecx
  __int64 v18; // rax
  __int64 v19; // r9
  __int64 v20; // rcx
  int v21; // eax
  int v22; // ecx
  __int64 v23; // rsi
  unsigned int v24; // eax
  _QWORD *v25; // r15
  __int64 v26; // rdx
  __int64 v27; // rax
  unsigned __int64 *v28; // rax
  unsigned __int64 v29; // r12
  __int64 v30; // rax
  _QWORD *v31; // r14
  __int64 v32; // rax
  __int64 v33; // rax
  bool v34; // zf
  __int64 v35; // rdx
  bool v36; // al
  unsigned __int64 v37; // rax
  __int64 *v38; // r12
  __int64 v39; // r14
  _BYTE *v40; // r15
  void *v41; // rax
  unsigned int v42; // esi
  int v43; // eax
  __int64 v44; // rdx
  _QWORD *v45; // r14
  int v46; // ecx
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 v49; // r9
  unsigned int v50; // ecx
  _QWORD *v51; // rax
  __int64 v52; // rdi
  _QWORD *v53; // rdx
  int v54; // r8d
  int v55; // eax
  int v56; // eax
  int v57; // eax
  __int64 v58; // r9
  _QWORD *v59; // rdi
  int v60; // r10d
  __int64 v61; // rcx
  __int64 v62; // rsi
  int v63; // eax
  __int64 v64; // r9
  int v65; // r10d
  unsigned int v66; // ecx
  __int64 v67; // rsi
  int v68; // edi
  int v69; // ecx
  __int64 v70; // rdi
  __int64 v71; // r8
  int v72; // r9d
  unsigned int v73; // eax
  __int64 v74; // rsi
  int v75; // r10d
  int v76; // eax
  int v77; // eax
  int v78; // ecx
  __int64 v79; // rdi
  int v80; // r9d
  unsigned int v81; // eax
  __int64 v82; // rsi
  __int64 v83; // [rsp+0h] [rbp-D0h]
  __int64 v84; // [rsp+8h] [rbp-C8h]
  __int64 v85; // [rsp+10h] [rbp-C0h]
  unsigned __int64 *v86; // [rsp+18h] [rbp-B8h]
  __int64 v87; // [rsp+20h] [rbp-B0h]
  __int64 v88; // [rsp+28h] [rbp-A8h]
  __int64 *v89; // [rsp+28h] [rbp-A8h]
  __int64 v90; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v91; // [rsp+38h] [rbp-98h] BYREF
  unsigned int v92; // [rsp+40h] [rbp-90h]
  _QWORD v93[2]; // [rsp+50h] [rbp-80h] BYREF
  __int64 v94; // [rsp+60h] [rbp-70h]
  __int64 (__fastcall **v95)(); // [rsp+70h] [rbp-60h] BYREF
  __int64 v96; // [rsp+78h] [rbp-58h] BYREF
  __int64 v97; // [rsp+80h] [rbp-50h]
  __int64 v98; // [rsp+88h] [rbp-48h]
  __int64 v99; // [rsp+90h] [rbp-40h]

  v2 = a1 + 8;
  v3 = a2;
  v97 = a2;
  v95 = 0;
  v96 = 0;
  v5 = a2 != -8192 && a2 != -4096 && a2 != 0;
  if ( v5 )
    sub_BD73F0((__int64)&v95);
  v6 = *(_DWORD *)(a1 + 32);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 8);
LABEL_5:
    sub_31E01F0(v2, 2 * v6);
    v7 = *(_DWORD *)(a1 + 32);
    if ( !v7 )
    {
LABEL_6:
      v8 = v97;
      v9 = 0;
LABEL_7:
      v10 = *(_DWORD *)(a1 + 24) + 1;
      goto LABEL_8;
    }
    v8 = v97;
    v69 = v7 - 1;
    v70 = *(_QWORD *)(a1 + 16);
    v71 = 0;
    v72 = 1;
    v73 = (v7 - 1) & (((unsigned int)v97 >> 9) ^ ((unsigned int)v97 >> 4));
    v9 = v70 + 48LL * v73;
    v74 = *(_QWORD *)(v9 + 16);
    if ( v74 == v97 )
      goto LABEL_7;
    while ( v74 != -4096 )
    {
      if ( v74 == -8192 && !v71 )
        v71 = v9;
      v73 = v69 & (v72 + v73);
      v9 = v70 + 48LL * v73;
      v74 = *(_QWORD *)(v9 + 16);
      if ( v97 == v74 )
        goto LABEL_7;
      ++v72;
    }
LABEL_126:
    if ( v71 )
      v9 = v71;
    goto LABEL_7;
  }
  v8 = v97;
  v16 = *(_QWORD *)(a1 + 16);
  v17 = (v6 - 1) & (((unsigned int)v97 >> 9) ^ ((unsigned int)v97 >> 4));
  v18 = v16 + 48LL * v17;
  v19 = *(_QWORD *)(v18 + 16);
  if ( v97 == v19 )
  {
LABEL_18:
    v15 = *(_QWORD *)(v18 + 32);
    v12 = (__int64 *)(v18 + 24);
    v14 = *(_DWORD *)(v18 + 40);
    v13 = 40LL * v14;
    goto LABEL_19;
  }
  v75 = 1;
  v9 = 0;
  while ( v19 != -4096 )
  {
    if ( v19 == -8192 && !v9 )
      v9 = v18;
    v17 = (v6 - 1) & (v75 + v17);
    v18 = v16 + 48LL * v17;
    v19 = *(_QWORD *)(v18 + 16);
    if ( v97 == v19 )
      goto LABEL_18;
    ++v75;
  }
  if ( !v9 )
    v9 = v18;
  v76 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  v10 = v76 + 1;
  if ( 4 * (v76 + 1) >= 3 * v6 )
    goto LABEL_5;
  if ( v6 - *(_DWORD *)(a1 + 28) - v10 <= v6 >> 3 )
  {
    sub_31E01F0(v2, v6);
    v77 = *(_DWORD *)(a1 + 32);
    if ( !v77 )
      goto LABEL_6;
    v8 = v97;
    v78 = v77 - 1;
    v79 = *(_QWORD *)(a1 + 16);
    v71 = 0;
    v80 = 1;
    v81 = (v77 - 1) & (((unsigned int)v97 >> 9) ^ ((unsigned int)v97 >> 4));
    v9 = v79 + 48LL * v81;
    v82 = *(_QWORD *)(v9 + 16);
    if ( v82 == v97 )
      goto LABEL_7;
    while ( v82 != -4096 )
    {
      if ( !v71 && v82 == -8192 )
        v71 = v9;
      v81 = v78 & (v80 + v81);
      v9 = v79 + 48LL * v81;
      v82 = *(_QWORD *)(v9 + 16);
      if ( v97 == v82 )
        goto LABEL_7;
      ++v80;
    }
    goto LABEL_126;
  }
LABEL_8:
  *(_DWORD *)(a1 + 24) = v10;
  if ( *(_QWORD *)(v9 + 16) == -4096 )
  {
    if ( v8 != -4096 )
    {
LABEL_13:
      *(_QWORD *)(v9 + 16) = v8;
      if ( v8 != -4096 && v8 != 0 && v8 != -8192 )
        sub_BD73F0(v9);
    }
  }
  else
  {
    --*(_DWORD *)(a1 + 28);
    v11 = *(_QWORD *)(v9 + 16);
    if ( v11 != v8 )
    {
      if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
      {
        v88 = v8;
        sub_BD60C0((_QWORD *)v9);
        v8 = v88;
      }
      goto LABEL_13;
    }
  }
  v12 = (__int64 *)(v9 + 24);
  v13 = 0;
  v14 = 0;
  *(_QWORD *)(v9 + 40) = 0;
  *(_OWORD *)(v9 + 24) = 0;
  v15 = 0;
LABEL_19:
  v20 = *v12;
  v92 = v14;
  *v12 = 0;
  v90 = v20;
  v91 = v15;
  if ( v97 != -4096 && v97 != 0 && v97 != -8192 )
    sub_BD60C0(&v95);
  v93[0] = 0;
  v93[1] = 0;
  v94 = v3;
  if ( v5 )
  {
    sub_BD73F0((__int64)v93);
    v3 = v94;
  }
  v21 = *(_DWORD *)(a1 + 32);
  if ( v21 )
  {
    v22 = v21 - 1;
    v23 = *(_QWORD *)(a1 + 16);
    v24 = (v21 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v25 = (_QWORD *)(v23 + 48LL * v24);
    v26 = v25[2];
    if ( v26 == v3 )
    {
LABEL_26:
      v27 = v25[3];
      if ( v27 )
      {
        if ( (v27 & 4) != 0 )
        {
          v28 = (unsigned __int64 *)(v27 & 0xFFFFFFFFFFFFFFF8LL);
          v29 = (unsigned __int64)v28;
          if ( v28 )
          {
            if ( (unsigned __int64 *)*v28 != v28 + 2 )
              _libc_free(*v28);
            j_j___libc_free_0(v29);
          }
        }
      }
      v95 = 0;
      v96 = 0;
      v97 = -8192;
      v30 = v25[2];
      if ( v30 != -8192 )
      {
        if ( v30 != -4096 && v30 )
          sub_BD60C0(v25);
        v25[2] = -8192;
        if ( v97 != -4096 && v97 != 0 && v97 != -8192 )
          sub_BD60C0(&v95);
      }
      --*(_DWORD *)(a1 + 24);
      v3 = v94;
      ++*(_DWORD *)(a1 + 28);
    }
    else
    {
      v68 = 1;
      while ( v26 != -4096 )
      {
        v24 = v22 & (v68 + v24);
        v25 = (_QWORD *)(v23 + 48LL * v24);
        v26 = v25[2];
        if ( v26 == v3 )
          goto LABEL_26;
        ++v68;
      }
    }
  }
  if ( v3 != 0 && v3 != -4096 && v3 != -8192 )
    sub_BD60C0(v93);
  v31 = (_QWORD *)(*(_QWORD *)(a1 + 40) + v13);
  v96 = 2;
  v97 = 0;
  v98 = 0;
  v95 = off_4A35038;
  v99 = 0;
  v32 = v31[3];
  if ( !v32 )
  {
    v31[4] = 0;
    goto LABEL_51;
  }
  if ( v32 == -8192 || v32 == -4096 )
  {
    v31[3] = 0;
    v36 = v98 != -8192 && v98 != -4096 && v98 != 0;
    v35 = 0;
  }
  else
  {
    sub_BD60C0(v31 + 1);
    v33 = v98;
    v34 = v98 == 0;
    v31[3] = v98;
    if ( v33 == -4096 || v34 || v33 == -8192 )
    {
      v31[4] = v99;
      goto LABEL_51;
    }
    sub_BD6050(v31 + 1, v96 & 0xFFFFFFFFFFFFFFF8LL);
    v35 = v99;
    v36 = v98 != -4096 && v98 != 0 && v98 != -8192;
  }
  v31[4] = v35;
  v95 = (__int64 (__fastcall **)())&unk_49DB368;
  if ( v36 )
    sub_BD60C0(&v96);
LABEL_51:
  v87 = v90;
  v83 = v90 >> 2;
  v37 = v90 & 0xFFFFFFFFFFFFFFF8LL;
  v86 = (unsigned __int64 *)(v90 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v90 & 4) != 0 )
  {
    v38 = *(__int64 **)v37;
    v89 = (__int64 *)(*(_QWORD *)v37 + 8LL * *(unsigned int *)(v37 + 8));
    if ( *(__int64 **)v37 != v89 )
      goto LABEL_54;
  }
  else if ( v86 )
  {
    v38 = &v90;
    v89 = &v91;
LABEL_54:
    v84 = a1 + 64;
    while ( 1 )
    {
      v39 = *v38;
      v93[0] = v39;
      v40 = *(_BYTE **)v39;
      if ( *(_QWORD *)v39 )
        goto LABEL_56;
      if ( (*(_BYTE *)(v39 + 9) & 0x70) == 0x20 && *(char *)(v39 + 8) >= 0 )
      {
        *(_BYTE *)(v39 + 8) |= 8u;
        v41 = sub_E807D0(*(_QWORD *)(v39 + 24));
        *(_QWORD *)v39 = v41;
        if ( v41 )
          goto LABEL_56;
      }
      v95 = 0;
      v96 = 0;
      v97 = v15;
      if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
        sub_BD73F0((__int64)&v95);
      v42 = *(_DWORD *)(a1 + 88);
      if ( !v42 )
        break;
      v44 = v97;
      v49 = *(_QWORD *)(a1 + 72);
      v50 = (v42 - 1) & (((unsigned int)v97 >> 9) ^ ((unsigned int)v97 >> 4));
      v51 = (_QWORD *)(v49 + 48LL * v50);
      v52 = v51[2];
      if ( v52 != v97 )
      {
        v45 = 0;
        v54 = 1;
        while ( v52 != -4096 )
        {
          if ( !v45 && v52 == -8192 )
            v45 = v51;
          v50 = (v42 - 1) & (v54 + v50);
          v51 = (_QWORD *)(v49 + 48LL * v50);
          v52 = v51[2];
          if ( v97 == v52 )
            goto LABEL_90;
          ++v54;
        }
        if ( !v45 )
          v45 = v51;
        v55 = *(_DWORD *)(a1 + 80);
        ++*(_QWORD *)(a1 + 64);
        v46 = v55 + 1;
        if ( 4 * (v55 + 1) < 3 * v42 )
        {
          if ( v42 - *(_DWORD *)(a1 + 84) - v46 > v42 >> 3 )
            goto LABEL_73;
          sub_31E3060(v84, v42);
          v56 = *(_DWORD *)(a1 + 88);
          if ( v56 )
          {
            v44 = v97;
            v57 = v56 - 1;
            v58 = *(_QWORD *)(a1 + 72);
            v59 = 0;
            v60 = 1;
            LODWORD(v61) = v57 & (((unsigned int)v97 >> 9) ^ ((unsigned int)v97 >> 4));
            v45 = (_QWORD *)(v58 + 48LL * (unsigned int)v61);
            v62 = v45[2];
            if ( v62 != v97 )
            {
              while ( v62 != -4096 )
              {
                if ( !v59 && v62 == -8192 )
                  v59 = v45;
                v61 = v57 & (unsigned int)(v61 + v60);
                v45 = (_QWORD *)(v58 + 48 * v61);
                v62 = v45[2];
                if ( v97 == v62 )
                  goto LABEL_72;
                ++v60;
              }
LABEL_117:
              if ( v59 )
                v45 = v59;
            }
LABEL_72:
            v46 = *(_DWORD *)(a1 + 80) + 1;
LABEL_73:
            *(_DWORD *)(a1 + 80) = v46;
            if ( v45[2] == -4096 )
            {
              if ( v44 != -4096 )
                goto LABEL_78;
            }
            else
            {
              --*(_DWORD *)(a1 + 84);
              v47 = v45[2];
              if ( v47 != v44 )
              {
                if ( v47 != 0 && v47 != -4096 && v47 != -8192 )
                {
                  v85 = v44;
                  sub_BD60C0(v45);
                  v44 = v85;
                }
LABEL_78:
                v45[2] = v44;
                if ( v44 != -4096 && v44 != 0 && v44 != -8192 )
                  sub_BD73F0((__int64)v45);
              }
            }
            v45[3] = 0;
            v48 = (__int64)(v45 + 3);
            v45[4] = 0;
            v45[5] = 0;
            goto LABEL_82;
          }
LABEL_71:
          v44 = v97;
          v45 = 0;
          goto LABEL_72;
        }
LABEL_70:
        sub_31E3060(v84, 2 * v42);
        v43 = *(_DWORD *)(a1 + 88);
        if ( v43 )
        {
          v44 = v97;
          v63 = v43 - 1;
          v64 = *(_QWORD *)(a1 + 72);
          v59 = 0;
          v65 = 1;
          v66 = v63 & (((unsigned int)v97 >> 9) ^ ((unsigned int)v97 >> 4));
          v45 = (_QWORD *)(v64 + 48LL * v66);
          v67 = v45[2];
          if ( v67 != v97 )
          {
            while ( v67 != -4096 )
            {
              if ( !v59 && v67 == -8192 )
                v59 = v45;
              v66 = v63 & (v65 + v66);
              v45 = (_QWORD *)(v64 + 48LL * v66);
              v67 = v45[2];
              if ( v97 == v67 )
                goto LABEL_72;
              ++v65;
            }
            goto LABEL_117;
          }
          goto LABEL_72;
        }
        goto LABEL_71;
      }
LABEL_90:
      v53 = (_QWORD *)v51[4];
      v40 = (_BYTE *)v51[5];
      v48 = (__int64)(v51 + 3);
      if ( v53 != (_QWORD *)v40 )
      {
        if ( v53 )
        {
          *v53 = v93[0];
          v53 = (_QWORD *)v51[4];
        }
        v51[4] = v53 + 1;
        goto LABEL_83;
      }
LABEL_82:
      sub_31DFB90(v48, v40, v93);
LABEL_83:
      if ( v97 != -4096 && v97 != 0 && v97 != -8192 )
        sub_BD60C0(&v95);
      if ( v89 == ++v38 )
      {
        if ( v87 )
          goto LABEL_57;
        return;
      }
    }
    ++*(_QWORD *)(a1 + 64);
    goto LABEL_70;
  }
LABEL_56:
  if ( v87 )
  {
LABEL_57:
    if ( (v83 & 1) != 0 && v86 )
    {
      if ( (unsigned __int64 *)*v86 != v86 + 2 )
        _libc_free(*v86);
      j_j___libc_free_0((unsigned __int64)v86);
    }
  }
}
