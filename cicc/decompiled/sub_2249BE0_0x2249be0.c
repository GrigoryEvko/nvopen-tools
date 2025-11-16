// Function: sub_2249BE0
// Address: 0x2249be0
//
_QWORD *__fastcall sub_2249BE0(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        __int64 *a8)
{
  __int64 v8; // r15
  bool v9; // bp
  char v10; // al
  wchar_t v11; // r12d
  char v12; // r13
  char v13; // bp
  char v14; // bp
  __int64 v15; // r14
  __int64 v16; // rax
  _QWORD *v17; // rdi
  unsigned __int64 v18; // rax
  int *v19; // rax
  int v20; // eax
  char v21; // r12
  char v22; // al
  char v23; // bp
  wchar_t *v24; // rax
  wchar_t v25; // eax
  unsigned __int64 v26; // rax
  wchar_t v27; // eax
  char v28; // si
  char v29; // al
  wchar_t *v30; // rax
  char v31; // bp
  __int64 v32; // r13
  __int64 v33; // rax
  _QWORD *v34; // rdi
  unsigned __int64 v35; // rax
  int *v36; // rax
  int v37; // eax
  char v38; // bp
  char v39; // al
  char v40; // r12
  __int64 v41; // rax
  __int64 v42; // rbx
  _QWORD *result; // rax
  bool v44; // zf
  __int64 v45; // rax
  __int64 v46; // rbp
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // r12
  __int64 v50; // rbp
  __int64 v51; // rax
  __int64 v52; // r12
  __int64 v53; // rax
  unsigned __int64 v54; // rax
  int v55; // edx
  _DWORD *v56; // rax
  int v57; // edx
  char v58; // si
  _DWORD *v59; // rax
  int v60; // edx
  wchar_t *v61; // rax
  wchar_t v62; // eax
  wchar_t *v63; // rax
  wchar_t v64; // eax
  wchar_t v65; // eax
  int v66; // edx
  int v67; // r14d
  _QWORD *v68; // rdi
  unsigned __int64 v69; // rax
  int *v70; // rax
  int v71; // eax
  char v72; // r13
  char v73; // al
  char v74; // dl
  bool v75; // al
  _BYTE *v76; // rax
  __int64 v77; // rbp
  __int64 v78; // rax
  int *v79; // rax
  int v80; // ecx
  _DWORD *v81; // rax
  int v82; // eax
  int v83; // ecx
  int v84; // edx
  _QWORD *v85; // [rsp+0h] [rbp-A8h]
  bool v86; // [rsp+10h] [rbp-98h]
  char v87; // [rsp+18h] [rbp-90h]
  int v88; // [rsp+20h] [rbp-88h]
  char v89; // [rsp+20h] [rbp-88h]
  __int16 v90; // [rsp+26h] [rbp-82h]
  wchar_t *s; // [rsp+28h] [rbp-80h]
  _QWORD *v92; // [rsp+40h] [rbp-68h] BYREF
  __int64 v93; // [rsp+48h] [rbp-60h]
  _QWORD *v94; // [rsp+50h] [rbp-58h] BYREF
  __int64 v95; // [rsp+58h] [rbp-50h]
  char v96; // [rsp+66h] [rbp-42h] BYREF
  __int64 v97[8]; // [rsp+68h] [rbp-40h] BYREF

  v94 = a2;
  v95 = a3;
  v92 = a4;
  v93 = a5;
  v8 = sub_22462F0((__int64)&v96, (__int64 *)(a6 + 208));
  v9 = sub_2247850((__int64)&v94, (__int64)&v92);
  if ( v9 )
  {
    v10 = *(_BYTE *)(v8 + 32);
    v11 = 0;
    v12 = 0;
    v88 = 0;
    goto LABEL_3;
  }
  v65 = sub_2247910((__int64)&v94);
  v66 = *(_DWORD *)(v8 + 228);
  v11 = v65;
  if ( v66 == v65 || *(_DWORD *)(v8 + 224) == v65 )
  {
    v10 = *(_BYTE *)(v8 + 32);
    if ( v10 && *(_DWORD *)(v8 + 76) == v11 || *(_DWORD *)(v8 + 72) == v11 )
      goto LABEL_136;
    sub_2215DF0(a8, 2 * (v66 != v11) + 43);
    sub_2240940(v94);
    LODWORD(v95) = -1;
    v75 = sub_2247850((__int64)&v94, (__int64)&v92);
    if ( v75 )
    {
      v88 = 0;
      v12 = 0;
      v9 = v75;
      v10 = *(_BYTE *)(v8 + 32);
      goto LABEL_3;
    }
    v11 = sub_2247910((__int64)&v94);
  }
  v10 = *(_BYTE *)(v8 + 32);
LABEL_136:
  v86 = v9;
  v12 = 0;
  v67 = 0;
  while ( 1 )
  {
    if ( v10 && *(_DWORD *)(v8 + 76) == v11 )
    {
      v9 = v86;
      v88 = v67;
      v97[0] = (__int64)&unk_4FD67D8;
      goto LABEL_132;
    }
    if ( *(_DWORD *)(v8 + 72) == v11 || *(_DWORD *)(v8 + 240) != v11 )
      break;
    if ( !v12 )
    {
      v77 = *(_QWORD *)(*a8 - 24);
      if ( (unsigned __int64)(v77 + 1) > *(_QWORD *)(*a8 - 16) || *(int *)(*a8 - 8) > 0 )
        sub_2215AB0(a8, v77 + 1);
      *(_BYTE *)(*a8 + *(_QWORD *)(*a8 - 24)) = 48;
      v78 = *a8;
      if ( (_UNKNOWN *)(*a8 - 24) != &unk_4FD67C0 )
      {
        *(_DWORD *)(v78 - 8) = 0;
        *(_QWORD *)(v78 - 24) = v77 + 1;
        *(_BYTE *)(v78 + v77 + 1) = 0;
      }
    }
    v68 = v94;
    ++v67;
    v69 = v94[2];
    if ( v69 >= v94[3] )
    {
      (*(void (__fastcall **)(_QWORD *))(*v94 + 80LL))(v94);
      v68 = v94;
      LODWORD(v95) = -1;
      if ( !v94 )
      {
        v72 = 1;
        goto LABEL_148;
      }
    }
    else
    {
      LODWORD(v95) = -1;
      v94[2] = v69 + 4;
    }
    v70 = (int *)v68[2];
    if ( (unsigned __int64)v70 >= v68[3] )
      v71 = (*(__int64 (__fastcall **)(_QWORD *))(*v68 + 72LL))(v68);
    else
      v71 = *v70;
    v72 = 0;
    if ( v71 == -1 )
    {
      v94 = 0;
      v72 = 1;
    }
LABEL_148:
    v73 = (_DWORD)v93 == -1;
    v74 = v73 & (v92 != 0);
    if ( v74 )
    {
      v79 = (int *)v92[2];
      if ( (unsigned __int64)v79 >= v92[3] )
      {
        v89 = v74;
        v82 = (*(__int64 (**)(void))(*v92 + 72LL))();
        v74 = v89;
        v80 = v82;
      }
      else
      {
        v80 = *v79;
      }
      v73 = 0;
      if ( v80 == -1 )
      {
        v92 = 0;
        v73 = v74;
      }
    }
    if ( v72 == v73 )
    {
      v88 = v67;
      v10 = *(_BYTE *)(v8 + 32);
      v12 = 1;
      v9 = 1;
      goto LABEL_3;
    }
    v11 = v95;
    if ( (_DWORD)v95 == -1 && v94 )
    {
      v81 = (_DWORD *)v94[2];
      v11 = (unsigned __int64)v81 >= v94[3] ? (*(__int64 (**)(void))(*v94 + 72LL))() : *v81;
      if ( v11 == -1 )
        v94 = 0;
    }
    v10 = *(_BYTE *)(v8 + 32);
    v12 = 1;
  }
  v88 = v67;
  v9 = v86;
LABEL_3:
  v97[0] = (__int64)&unk_4FD67D8;
  if ( v10 )
LABEL_132:
    sub_2215AB0(v97, 0x20u);
  v87 = *(_BYTE *)(v8 + 328);
  if ( !v87 )
  {
    if ( !v9 )
    {
      v90 = 0;
LABEL_7:
      v13 = v11 - 48;
      if ( (unsigned int)(v11 - 48) <= 9 )
      {
LABEL_8:
        v14 = v13 + 48;
        v15 = *(_QWORD *)(*a8 - 24);
        if ( (unsigned __int64)(v15 + 1) > *(_QWORD *)(*a8 - 16) || *(int *)(*a8 - 8) > 0 )
          sub_2215AB0(a8, v15 + 1);
        v12 = 1;
        *(_BYTE *)(*a8 + *(_QWORD *)(*a8 - 24)) = v14;
        v16 = *a8;
        if ( (_UNKNOWN *)(*a8 - 24) != &unk_4FD67C0 )
        {
          *(_DWORD *)(v16 - 8) = 0;
          *(_QWORD *)(v16 - 24) = v15 + 1;
          *(_BYTE *)(v16 + v15 + 1) = 0;
        }
        goto LABEL_13;
      }
      while ( 1 )
      {
        if ( *(_DWORD *)(v8 + 72) == v11 && !v90 )
        {
          sub_2215DF0(a8, 46);
          v90 = 256;
LABEL_13:
          v17 = v94;
          v18 = v94[2];
          if ( v18 >= v94[3] )
            goto LABEL_37;
LABEL_14:
          LODWORD(v95) = -1;
          v17[2] = v18 + 4;
LABEL_15:
          v19 = (int *)v17[2];
          if ( (unsigned __int64)v19 >= v17[3] )
            v20 = (*(__int64 (__fastcall **)(_QWORD *))(*v17 + 72LL))(v17);
          else
            v20 = *v19;
          v21 = 0;
          if ( v20 == -1 )
          {
            v94 = 0;
            v21 = 1;
          }
          goto LABEL_19;
        }
        if ( *(_DWORD *)(v8 + 296) != v11 && *(_DWORD *)(v8 + 320) != v11 )
          goto LABEL_59;
        v12 &= v90 ^ 1;
        if ( !v12 )
          goto LABEL_59;
        sub_2215DF0(a8, 101);
        v26 = v94[2];
        if ( v26 >= v94[3] )
          (*(void (__fastcall **)(_QWORD *))(*v94 + 80LL))(v94);
        else
          v94[2] = v26 + 4;
        LODWORD(v95) = -1;
        if ( sub_2247850((__int64)&v94, (__int64)&v92) )
          goto LABEL_200;
        v27 = sub_2247910((__int64)&v94);
        v11 = v27;
        if ( *(_DWORD *)(v8 + 228) == v27 )
        {
          v28 = 43;
        }
        else
        {
          v28 = 45;
          if ( *(_DWORD *)(v8 + 224) != v27 )
          {
            LOBYTE(v90) = v12;
            goto LABEL_7;
          }
        }
        sub_2215DF0(a8, v28);
        v17 = v94;
        LOBYTE(v90) = v12;
        v18 = v94[2];
        if ( v18 < v94[3] )
          goto LABEL_14;
LABEL_37:
        (*(void (__fastcall **)(_QWORD *))(*v17 + 80LL))(v17);
        v17 = v94;
        LODWORD(v95) = -1;
        if ( v94 )
          goto LABEL_15;
        v21 = 1;
LABEL_19:
        v22 = (_DWORD)v93 == -1;
        v23 = v22 & (v92 != 0);
        if ( v23 )
        {
          v56 = (_DWORD *)v92[2];
          v57 = (unsigned __int64)v56 >= v92[3] ? (*(__int64 (**)(void))(*v92 + 72LL))() : *v56;
          v22 = 0;
          if ( v57 == -1 )
          {
            v92 = 0;
            v22 = v23;
          }
        }
        if ( v22 == v21 )
          goto LABEL_59;
        v11 = v95;
        if ( (_DWORD)v95 != -1 || !v94 )
          goto LABEL_7;
        v24 = (wchar_t *)v94[2];
        if ( (unsigned __int64)v24 >= v94[3] )
        {
          v25 = (*(__int64 (**)(void))(*v94 + 72LL))();
          v11 = v25;
        }
        else
        {
          v11 = *v24;
          v25 = *v24;
        }
        if ( v25 != -1 )
          goto LABEL_7;
        v94 = 0;
        v13 = v11 - 48;
        if ( (unsigned int)(v11 - 48) <= 9 )
          goto LABEL_8;
      }
    }
LABEL_165:
    v42 = v97[0];
    if ( *(_QWORD *)(v97[0] - 24) )
      goto LABEL_62;
    goto LABEL_65;
  }
  if ( v9 )
    goto LABEL_165;
  v90 = 0;
  s = (wchar_t *)(v8 + 240);
  v29 = *(_BYTE *)(v8 + 32);
  while ( 2 )
  {
    while ( 2 )
    {
      if ( v29 && *(_DWORD *)(v8 + 76) == v11 )
      {
        v44 = v90 == 0;
        HIBYTE(v90) |= v90;
        if ( !v44 )
        {
          v45 = v97[0];
LABEL_197:
          v42 = v45;
          if ( *(_QWORD *)(v45 - 24) )
            goto LABEL_63;
          goto LABEL_65;
        }
        if ( !v88 )
        {
          v76 = (_BYTE *)*a8;
          if ( *(int *)(*a8 - 8) <= 0 )
          {
            if ( v76 - 24 != (_BYTE *)&unk_4FD67C0 )
            {
              *((_DWORD *)v76 - 2) = 0;
              *((_QWORD *)v76 - 3) = 0;
              *v76 = 0;
            }
          }
          else
          {
            if ( v76 - 24 != (_BYTE *)&unk_4FD67C0 )
            {
              if ( &_pthread_key_create )
              {
                v84 = _InterlockedExchangeAdd((volatile signed __int32 *)v76 - 2, 0xFFFFFFFF);
              }
              else
              {
                v84 = *((_DWORD *)v76 - 2);
                *((_DWORD *)v76 - 2) = v84 - 1;
              }
              if ( v84 <= 0 )
                j_j___libc_free_0_1((unsigned __int64)(v76 - 24));
            }
            *a8 = (__int64)&unk_4FD67D8;
          }
          goto LABEL_165;
        }
        v49 = *(_QWORD *)(v97[0] - 24);
        v50 = v49 + 1;
        if ( (unsigned __int64)(v49 + 1) > *(_QWORD *)(v97[0] - 16) || *(int *)(v97[0] - 8) > 0 )
          sub_2215AB0(v97, v49 + 1);
        v58 = v88;
        LOBYTE(v90) = 0;
        v88 = 0;
        *(_BYTE *)(v97[0] + *(_QWORD *)(v97[0] - 24)) = v58;
        v51 = v97[0];
        if ( (_UNKNOWN *)(v97[0] - 24) == &unk_4FD67C0 )
          break;
LABEL_79:
        *(_DWORD *)(v51 - 8) = 0;
        *(_QWORD *)(v51 - 24) = v50;
        *(_BYTE *)(v51 + v49 + 1) = 0;
        v34 = v94;
        v35 = v94[2];
        if ( v35 < v94[3] )
          goto LABEL_52;
        goto LABEL_80;
      }
      if ( *(_DWORD *)(v8 + 72) == v11 )
      {
        v44 = v90 == 0;
        LOBYTE(v90) = HIBYTE(v90) | v90;
        v45 = v97[0];
        v46 = *(_QWORD *)(v97[0] - 24);
        if ( !v44 )
          goto LABEL_197;
        if ( v46 )
        {
          if ( (unsigned __int64)(v46 + 1) > *(_QWORD *)(v97[0] - 16) || *(int *)(v97[0] - 8) > 0 )
            sub_2215AB0(v97, v46 + 1);
          *(_BYTE *)(v97[0] + *(_QWORD *)(v97[0] - 24)) = v88;
          v47 = v97[0];
          v48 = v97[0] - 24;
          if ( (_UNKNOWN *)(v97[0] - 24) != &unk_4FD67C0 )
          {
            *(_DWORD *)(v97[0] - 8) = 0;
            *(_QWORD *)(v47 - 24) = v46 + 1;
            *(_BYTE *)(v48 + v46 + 25) = 0;
          }
        }
        v49 = *(_QWORD *)(*a8 - 24);
        v50 = v49 + 1;
        if ( (unsigned __int64)(v49 + 1) > *(_QWORD *)(*a8 - 16) || *(int *)(*a8 - 8) > 0 )
          sub_2215AB0(a8, v49 + 1);
        HIBYTE(v90) = v87;
        *(_BYTE *)(*a8 + *(_QWORD *)(*a8 - 24)) = 46;
        v51 = *a8;
        if ( (_UNKNOWN *)(*a8 - 24) == &unk_4FD67C0 )
          break;
        goto LABEL_79;
      }
      v30 = wmemchr(s, v11, 0xAu);
      if ( v30 )
      {
        v31 = v30 - s + 48;
        v32 = *(_QWORD *)(*a8 - 24);
        if ( (unsigned __int64)(v32 + 1) > *(_QWORD *)(*a8 - 16) || *(int *)(*a8 - 8) > 0 )
          sub_2215AB0(a8, v32 + 1);
        *(_BYTE *)(*a8 + *(_QWORD *)(*a8 - 24)) = v31;
        v33 = *a8;
        if ( (_UNKNOWN *)(*a8 - 24) != &unk_4FD67C0 )
        {
          *(_DWORD *)(v33 - 8) = 0;
          *(_QWORD *)(v33 - 24) = v32 + 1;
          *(_BYTE *)(v33 + v32 + 1) = 0;
        }
        ++v88;
        v12 = v87;
      }
      else
      {
        if ( *(_DWORD *)(v8 + 296) != v11 && *(_DWORD *)(v8 + 320) != v11 )
          goto LABEL_59;
        v41 = v97[0];
        v12 &= v90 ^ 1;
        if ( !v12 )
          goto LABEL_60;
        if ( *(_QWORD *)(v97[0] - 24) && !HIBYTE(v90) )
          sub_2215DF0(v97, v88);
        v52 = *(_QWORD *)(*a8 - 24);
        if ( (unsigned __int64)(v52 + 1) > *(_QWORD *)(*a8 - 16) || *(int *)(*a8 - 8) > 0 )
          sub_2215AB0(a8, v52 + 1);
        *(_BYTE *)(*a8 + *(_QWORD *)(*a8 - 24)) = 101;
        v53 = *a8;
        if ( (_UNKNOWN *)(*a8 - 24) != &unk_4FD67C0 )
        {
          *(_DWORD *)(v53 - 8) = 0;
          *(_QWORD *)(v53 - 24) = v52 + 1;
          *(_BYTE *)(v53 + v52 + 1) = 0;
        }
        v54 = v94[2];
        if ( v54 >= v94[3] )
          (*(void (__fastcall **)(_QWORD *))(*v94 + 80LL))(v94);
        else
          v94[2] = v54 + 4;
        LODWORD(v95) = -1;
        if ( sub_2247850((__int64)&v94, (__int64)&v92) )
        {
LABEL_200:
          v42 = v97[0];
          if ( *(_QWORD *)(v97[0] - 24) )
            goto LABEL_63;
          goto LABEL_65;
        }
        v11 = v95;
        if ( (_DWORD)v95 == -1 && v94 )
        {
          v63 = (wchar_t *)v94[2];
          if ( (unsigned __int64)v63 >= v94[3] )
          {
            v64 = (*(__int64 (**)(void))(*v94 + 72LL))();
            v11 = v64;
          }
          else
          {
            v11 = *v63;
            v64 = *v63;
          }
          if ( v64 == -1 )
            v94 = 0;
        }
        v55 = *(_DWORD *)(v8 + 228);
        v29 = *(_BYTE *)(v8 + 32);
        if ( v55 != v11 && *(_DWORD *)(v8 + 224) != v11 )
          goto LABEL_103;
        if ( v29 && *(_DWORD *)(v8 + 76) == v11 )
        {
          LOBYTE(v90) = *(_BYTE *)(v8 + 32);
          v12 = v90;
          continue;
        }
        if ( *(_DWORD *)(v8 + 72) == v11 )
        {
LABEL_103:
          LOBYTE(v90) = v12;
          continue;
        }
        sub_2215DF0(a8, 2 * (v55 != v11) + 43);
        LOBYTE(v90) = v12;
      }
      break;
    }
    v34 = v94;
    v35 = v94[2];
    if ( v35 < v94[3] )
    {
LABEL_52:
      LODWORD(v95) = -1;
      v34[2] = v35 + 4;
LABEL_53:
      v36 = (int *)v34[2];
      if ( (unsigned __int64)v36 >= v34[3] )
        v37 = (*(__int64 (__fastcall **)(_QWORD *))(*v34 + 72LL))(v34);
      else
        v37 = *v36;
      v38 = 0;
      if ( v37 == -1 )
      {
        v94 = 0;
        v38 = v87;
      }
      goto LABEL_57;
    }
LABEL_80:
    (*(void (__fastcall **)(_QWORD *))(*v34 + 80LL))(v34);
    v34 = v94;
    LODWORD(v95) = -1;
    if ( v94 )
      goto LABEL_53;
    v38 = v87;
LABEL_57:
    v39 = (_DWORD)v93 == -1;
    v40 = v39 & (v92 != 0);
    if ( v40 )
    {
      v59 = (_DWORD *)v92[2];
      v60 = (unsigned __int64)v59 >= v92[3] ? (*(__int64 (**)(void))(*v92 + 72LL))() : *v59;
      v39 = 0;
      if ( v60 == -1 )
      {
        v92 = 0;
        v39 = v40;
      }
    }
    if ( v39 != v38 )
    {
      v11 = v95;
      if ( (_DWORD)v95 == -1 && v94 )
      {
        v61 = (wchar_t *)v94[2];
        if ( (unsigned __int64)v61 >= v94[3] )
        {
          v62 = (*(__int64 (**)(void))(*v94 + 72LL))();
          v11 = v62;
        }
        else
        {
          v11 = *v61;
          v62 = *v61;
        }
        if ( v62 == -1 )
          v94 = 0;
      }
      v29 = *(_BYTE *)(v8 + 32);
      continue;
    }
    break;
  }
LABEL_59:
  v41 = v97[0];
LABEL_60:
  v42 = v41;
  if ( !*(_QWORD *)(v41 - 24) )
    goto LABEL_65;
  if ( v90 )
    goto LABEL_63;
LABEL_62:
  sub_2215DF0(v97, v88);
  v42 = v97[0];
LABEL_63:
  if ( !(unsigned __int8)sub_2255280(*(_QWORD *)(v8 + 16), *(_QWORD *)(v8 + 24), v97) )
    *a7 = 4;
LABEL_65:
  result = v94;
  if ( (_UNKNOWN *)(v42 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v83 = _InterlockedExchangeAdd((volatile signed __int32 *)(v42 - 8), 0xFFFFFFFF);
    }
    else
    {
      v83 = *(_DWORD *)(v42 - 8);
      *(_DWORD *)(v42 - 8) = v83 - 1;
    }
    if ( v83 <= 0 )
    {
      v85 = result;
      j_j___libc_free_0_1(v42 - 24);
      return v85;
    }
  }
  return result;
}
