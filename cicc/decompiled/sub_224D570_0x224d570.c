// Function: sub_224D570
// Address: 0x224d570
//
_QWORD *__fastcall sub_224D570(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        unsigned __int64 *a8)
{
  __int64 v9; // rbx
  int v10; // r14d
  int v11; // r13d
  char v12; // r15
  wchar_t v13; // eax
  wchar_t v14; // ebp
  char v15; // cl
  bool v16; // r12
  int v17; // r15d
  _QWORD *v18; // rdi
  unsigned __int64 v19; // rax
  int *v20; // rax
  int v21; // eax
  char v22; // cl
  char v23; // al
  char v24; // dl
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // r14
  int v27; // ebp
  _QWORD *v28; // rdi
  unsigned __int64 v29; // rax
  int *v30; // rax
  int v31; // eax
  char v32; // r13
  char v33; // al
  char v34; // bp
  bool v35; // bp
  __int64 v36; // rdx
  _QWORD *v37; // r12
  bool v39; // al
  bool v40; // cf
  int *v41; // rax
  int v42; // esi
  wchar_t *v43; // rax
  wchar_t v44; // eax
  _DWORD *v45; // rax
  wchar_t *v46; // rax
  __int64 v47; // rax
  _QWORD *v48; // rdi
  unsigned __int64 v49; // rax
  wchar_t *v50; // rax
  wchar_t v51; // eax
  _DWORD *v52; // rax
  int v53; // eax
  int v54; // eax
  unsigned __int64 v55; // [rsp+0h] [rbp-A8h]
  int v56; // [rsp+8h] [rbp-A0h]
  __int64 v57; // [rsp+18h] [rbp-90h]
  size_t n; // [rsp+30h] [rbp-78h]
  int v59; // [rsp+38h] [rbp-70h]
  char v60; // [rsp+38h] [rbp-70h]
  char v61; // [rsp+3Dh] [rbp-6Bh]
  char v62; // [rsp+3Dh] [rbp-6Bh]
  char v63; // [rsp+3Eh] [rbp-6Ah]
  bool v64; // [rsp+3Fh] [rbp-69h]
  _QWORD *v65; // [rsp+40h] [rbp-68h] BYREF
  __int64 v66; // [rsp+48h] [rbp-60h]
  _QWORD *v67; // [rsp+50h] [rbp-58h] BYREF
  __int64 v68; // [rsp+58h] [rbp-50h]
  char v69; // [rsp+66h] [rbp-42h] BYREF
  __int64 v70[8]; // [rsp+68h] [rbp-40h] BYREF

  v67 = a2;
  v68 = a3;
  v65 = a4;
  v66 = a5;
  v9 = sub_22462F0((__int64)&v69, (__int64 *)(a6 + 208));
  v10 = *(_DWORD *)(a6 + 24) & 0x4A;
  if ( v10 == 64 )
  {
    v11 = 8;
  }
  else
  {
    v11 = 10;
    if ( v10 == 8 )
    {
      v11 = 16;
      v12 = sub_2247850((__int64)&v67, (__int64)&v65);
      if ( v12 )
      {
        v64 = 0;
        v15 = *(_BYTE *)(v9 + 32);
        v14 = 0;
        v16 = 0;
        n = 22;
        v59 = 0;
        v56 = 22;
        goto LABEL_38;
      }
      goto LABEL_4;
    }
  }
  v12 = sub_2247850((__int64)&v67, (__int64)&v65);
  if ( v12 )
  {
    v56 = v11;
    v15 = *(_BYTE *)(v9 + 32);
    v14 = 0;
    n = v11;
    v16 = 0;
    v59 = 0;
    v64 = 0;
    goto LABEL_38;
  }
LABEL_4:
  v13 = sub_2247910((__int64)&v67);
  v14 = v13;
  v64 = *(_DWORD *)(v9 + 224) == v13;
  if ( *(_DWORD *)(v9 + 224) == v13 || *(_DWORD *)(v9 + 228) == v13 )
  {
    v15 = *(_BYTE *)(v9 + 32);
    if ( v15 && *(_DWORD *)(v9 + 76) == v13 || *(_DWORD *)(v9 + 72) == v13 )
      goto LABEL_17;
    sub_2240940(v67);
    LODWORD(v68) = -1;
    v12 = sub_2247850((__int64)&v67, (__int64)&v65);
    if ( v12 )
    {
      v59 = 0;
      v15 = *(_BYTE *)(v9 + 32);
      v16 = 0;
      if ( v11 != 16 )
        goto LABEL_37;
LABEL_10:
      n = 22;
      v56 = 22;
      goto LABEL_38;
    }
    v14 = sub_2247910((__int64)&v67);
  }
  v15 = *(_BYTE *)(v9 + 32);
LABEL_17:
  v17 = 0;
  v16 = 0;
  while ( 1 )
  {
    if ( v15 && *(_DWORD *)(v9 + 76) == v14 || *(_DWORD *)(v9 + 72) == v14 )
    {
LABEL_73:
      v59 = v17;
      v12 = 0;
      goto LABEL_36;
    }
    if ( *(_DWORD *)(v9 + 240) != v14 )
      break;
    v39 = !v16 || v11 == 10;
    if ( !v39 )
      goto LABEL_23;
    v16 = v11 == 8 || v10 == 0;
    if ( !v16 )
    {
      ++v17;
      v16 = v39;
      goto LABEL_27;
    }
    v18 = v67;
    v17 = 0;
    v11 = 8;
    v19 = v67[2];
    if ( v19 < v67[3] )
    {
LABEL_28:
      LODWORD(v68) = -1;
      v18[2] = v19 + 4;
LABEL_29:
      v20 = (int *)v18[2];
      if ( (unsigned __int64)v20 >= v18[3] )
        v21 = (*(__int64 (__fastcall **)(_QWORD *))(*v18 + 72LL))(v18);
      else
        v21 = *v20;
      v22 = 0;
      if ( v21 == -1 )
      {
        v67 = 0;
        v22 = 1;
      }
      goto LABEL_33;
    }
LABEL_68:
    (*(void (__fastcall **)(_QWORD *))(*v18 + 80LL))(v18);
    v18 = v67;
    LODWORD(v68) = -1;
    if ( v67 )
      goto LABEL_29;
    v22 = 1;
LABEL_33:
    v23 = (_DWORD)v66 == -1;
    v24 = v23 & (v65 != 0);
    if ( v24 )
    {
      v41 = (int *)v65[2];
      if ( (unsigned __int64)v41 >= v65[3] )
      {
        v62 = v22;
        v60 = v24;
        v53 = (*(__int64 (**)(void))(*v65 + 72LL))();
        v22 = v62;
        v24 = v60;
        v42 = v53;
      }
      else
      {
        v42 = *v41;
      }
      v23 = 0;
      if ( v42 == -1 )
      {
        v65 = 0;
        v23 = v24;
      }
    }
    if ( v22 == v23 )
    {
      v15 = *(_BYTE *)(v9 + 32);
      v59 = v17;
      v12 = 1;
      goto LABEL_36;
    }
    v14 = v68;
    if ( v67 && (_DWORD)v68 == -1 )
    {
      v52 = (_DWORD *)v67[2];
      v14 = (unsigned __int64)v52 >= v67[3] ? (*(__int64 (**)(void))(*v67 + 72LL))() : *v52;
      if ( v14 == -1 )
        v67 = 0;
    }
    v15 = *(_BYTE *)(v9 + 32);
    if ( !v16 )
      goto LABEL_73;
  }
  if ( !v16 )
    goto LABEL_73;
LABEL_23:
  if ( *(_DWORD *)(v9 + 232) == v14 || *(_DWORD *)(v9 + 236) == v14 )
  {
    if ( v10 != 0 && v11 != 16 )
    {
      v59 = v17;
      v12 = 0;
      v16 = 1;
      n = v11;
      v56 = v11;
      goto LABEL_38;
    }
    v17 = 0;
    v16 = 0;
    v11 = 16;
LABEL_27:
    v18 = v67;
    v19 = v67[2];
    if ( v19 < v67[3] )
      goto LABEL_28;
    goto LABEL_68;
  }
  v59 = v17;
  v16 = 1;
  v12 = 0;
LABEL_36:
  if ( v11 == 16 )
    goto LABEL_10;
LABEL_37:
  v56 = v11;
  n = v11;
LABEL_38:
  v70[0] = (__int64)&unk_4FD67D8;
  if ( v15 )
    sub_2215AB0(v70, 0x20u);
  v25 = 0xFFFFFFFFFFFFFFFFLL % v11;
  v57 = v11;
  v55 = 0xFFFFFFFFFFFFFFFFLL / v11;
  v63 = *(_BYTE *)(v9 + 328);
  if ( !v63 )
  {
    v61 = 0;
    if ( v12 )
    {
      v35 = v12;
      v26 = 0;
      v12 = 0;
      goto LABEL_57;
    }
    v26 = 0;
    while ( 2 )
    {
      if ( v56 <= 10 )
      {
        if ( v14 <= 47 || v14 >= (int)n + 48 )
          goto LABEL_106;
        v27 = v14 - 48;
LABEL_47:
        if ( v55 < v26 )
          goto LABEL_48;
LABEL_76:
        v28 = v67;
        v25 = ~(__int64)v27;
        v40 = v25 < v26 * v57;
        v26 = v27 + v26 * v57;
        v29 = v67[2];
        LOBYTE(v25) = v40;
        ++v59;
        v61 |= v40;
        if ( v29 >= v67[3] )
          goto LABEL_77;
LABEL_49:
        LODWORD(v68) = -1;
        v28[2] = v29 + 4;
        goto LABEL_50;
      }
      if ( (unsigned int)(v14 - 48) > 9 )
      {
        if ( (unsigned int)(v14 - 97) > 5 )
        {
          if ( (unsigned int)(v14 - 65) > 5 )
            goto LABEL_106;
          v27 = v14 - 55;
        }
        else
        {
          v27 = v14 - 87;
        }
        goto LABEL_47;
      }
      v27 = v14 - 48;
      if ( v55 >= v26 )
        goto LABEL_76;
LABEL_48:
      v28 = v67;
      v61 = 1;
      v29 = v67[2];
      if ( v29 < v67[3] )
        goto LABEL_49;
LABEL_77:
      (*(void (__fastcall **)(_QWORD *, __int64 *, unsigned __int64))(*v28 + 80LL))(v28, v70, v25);
      v28 = v67;
      LODWORD(v68) = -1;
      if ( v67 )
      {
LABEL_50:
        v30 = (int *)v28[2];
        if ( (unsigned __int64)v30 >= v28[3] )
          v31 = (*(__int64 (__fastcall **)(_QWORD *, __int64 *, unsigned __int64))(*v28 + 72LL))(v28, v70, v25);
        else
          v31 = *v30;
        v32 = 0;
        if ( v31 == -1 )
        {
          v67 = 0;
          v32 = 1;
        }
      }
      else
      {
        v32 = 1;
      }
      v33 = (_DWORD)v66 == -1;
      v34 = v33 & (v65 != 0);
      if ( v34 )
      {
        v45 = (_DWORD *)v65[2];
        v25 = (unsigned __int64)v45 >= v65[3]
            ? (*(unsigned int (__fastcall **)(_QWORD *, __int64 *, unsigned __int64))(*v65 + 72LL))(v65, v70, v25)
            : (unsigned int)*v45;
        v33 = 0;
        if ( (_DWORD)v25 == -1 )
        {
          v65 = 0;
          v33 = v34;
        }
      }
      if ( v33 == v32 )
      {
        v35 = 1;
        goto LABEL_57;
      }
      v14 = v68;
      if ( (_DWORD)v68 == -1 && v67 )
      {
        v43 = (wchar_t *)v67[2];
        if ( (unsigned __int64)v43 >= v67[3] )
        {
          v44 = (*(__int64 (__fastcall **)(_QWORD *, __int64 *, unsigned __int64))(*v67 + 72LL))(v67, v70, v25);
          v14 = v44;
        }
        else
        {
          v14 = *v43;
          v44 = *v43;
        }
        if ( v44 == -1 )
          v67 = 0;
      }
      continue;
    }
  }
  if ( v12 )
  {
    v35 = v12;
    v61 = 0;
    v26 = 0;
    v12 = 0;
    goto LABEL_57;
  }
  v61 = 0;
  v26 = 0;
  while ( 2 )
  {
    if ( !*(_BYTE *)(v9 + 32) || *(_DWORD *)(v9 + 76) != v14 )
    {
      if ( v14 == *(_DWORD *)(v9 + 72) || (v46 = wmemchr((const wchar_t *)(v9 + 240), v14, n)) == 0 )
      {
LABEL_106:
        v35 = 0;
        goto LABEL_57;
      }
      v47 = ((__int64)v46 - v9 - 240) >> 2;
      if ( (int)v47 > 15 )
        LODWORD(v47) = v47 - 6;
      if ( v55 >= v26 )
      {
        v40 = ~(__int64)(int)v47 < v26 * v11;
        v26 = (int)v47 + v26 * v11;
        ++v59;
        v61 |= v40;
        goto LABEL_117;
      }
      v48 = v67;
      v61 = v63;
      v49 = v67[2];
      if ( v49 >= v67[3] )
      {
LABEL_127:
        (*(void (__fastcall **)(_QWORD *))(*v48 + 80LL))(v48);
        goto LABEL_119;
      }
LABEL_118:
      v48[2] = v49 + 4;
LABEL_119:
      LODWORD(v68) = -1;
      v35 = sub_2247850((__int64)&v67, (__int64)&v65);
      if ( v35 )
        goto LABEL_57;
      v14 = v68;
      if ( v67 && (_DWORD)v68 == -1 )
      {
        v50 = (wchar_t *)v67[2];
        if ( (unsigned __int64)v50 >= v67[3] )
        {
          v51 = (*(__int64 (**)(void))(*v67 + 72LL))();
          v14 = v51;
        }
        else
        {
          v14 = *v50;
          v51 = *v50;
        }
        if ( v51 == -1 )
          v67 = 0;
      }
      continue;
    }
    break;
  }
  if ( v59 )
  {
    sub_2215DF0(v70, v59);
    v59 = 0;
LABEL_117:
    v48 = v67;
    v49 = v67[2];
    if ( v49 >= v67[3] )
      goto LABEL_127;
    goto LABEL_118;
  }
  v35 = 0;
  v12 = *(_BYTE *)(v9 + 32);
LABEL_57:
  v36 = v70[0];
  if ( *(_QWORD *)(v70[0] - 24) )
  {
    sub_2215DF0(v70, v59);
    if ( !(unsigned __int8)sub_2255280(*(_QWORD *)(v9 + 16), *(_QWORD *)(v9 + 24), v70) )
      *a7 = 4;
    v36 = v70[0];
    if ( v59 || v16 || *(_QWORD *)(v70[0] - 24) )
      goto LABEL_96;
LABEL_60:
    *a8 = 0;
    *a7 = 4;
  }
  else
  {
    if ( !v16 && !v59 )
      goto LABEL_60;
LABEL_96:
    if ( v12 )
      goto LABEL_60;
    if ( v61 )
    {
      *a8 = -1;
      *a7 = 4;
    }
    else
    {
      if ( v64 )
        v26 = -(__int64)v26;
      *a8 = v26;
    }
  }
  if ( v35 )
    *a7 |= 2u;
  v37 = v67;
  if ( (_UNKNOWN *)(v36 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v54 = _InterlockedExchangeAdd((volatile signed __int32 *)(v36 - 8), 0xFFFFFFFF);
    }
    else
    {
      v54 = *(_DWORD *)(v36 - 8);
      *(_DWORD *)(v36 - 8) = v54 - 1;
    }
    if ( v54 <= 0 )
      j_j___libc_free_0_1(v36 - 24);
  }
  return v37;
}
