// Function: sub_224E040
// Address: 0x224e040
//
_QWORD *__fastcall sub_224E040(
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
  int v10; // r15d
  int v11; // r13d
  wchar_t v12; // eax
  wchar_t v13; // ebp
  char v14; // cl
  int v15; // r14d
  char v16; // r12
  unsigned __int64 v17; // rax
  _QWORD *v18; // rdi
  unsigned __int64 v19; // rax
  int *v20; // rax
  int v21; // eax
  char v22; // cl
  char v23; // al
  char v24; // dl
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // r15
  int v28; // ebp
  _QWORD *v29; // rdi
  unsigned __int64 v30; // rax
  int *v31; // rax
  int v32; // eax
  char v33; // r13
  char v34; // al
  char v35; // bp
  bool v36; // bp
  __int64 v37; // rdx
  _QWORD *v38; // r12
  char v40; // al
  bool v41; // cf
  int *v42; // rax
  int v43; // esi
  wchar_t *v44; // rax
  wchar_t v45; // eax
  unsigned __int64 v46; // rax
  _DWORD *v47; // rax
  char v48; // al
  wchar_t *v49; // rax
  __int64 v50; // rax
  _QWORD *v51; // rdi
  unsigned __int64 v52; // rax
  _DWORD *v53; // rax
  __int64 v54; // rax
  int v55; // eax
  int v56; // eax
  int v57; // [rsp+8h] [rbp-C0h]
  unsigned __int64 v58; // [rsp+10h] [rbp-B8h]
  unsigned __int64 v59; // [rsp+28h] [rbp-A0h]
  __int64 v60; // [rsp+30h] [rbp-98h]
  size_t n; // [rsp+40h] [rbp-88h]
  unsigned __int64 v62; // [rsp+48h] [rbp-80h]
  char v63; // [rsp+55h] [rbp-73h]
  char v64; // [rsp+55h] [rbp-73h]
  char v65; // [rsp+56h] [rbp-72h]
  char v66; // [rsp+56h] [rbp-72h]
  bool v67; // [rsp+57h] [rbp-71h]
  _QWORD *v68; // [rsp+60h] [rbp-68h] BYREF
  __int64 v69; // [rsp+68h] [rbp-60h]
  _QWORD *v70; // [rsp+70h] [rbp-58h] BYREF
  __int64 v71; // [rsp+78h] [rbp-50h]
  char v72; // [rsp+86h] [rbp-42h] BYREF
  __int64 v73[8]; // [rsp+88h] [rbp-40h] BYREF

  v70 = a2;
  v71 = a3;
  v68 = a4;
  v69 = a5;
  v9 = sub_22462F0((__int64)&v72, (__int64 *)(a6 + 208));
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
      if ( !sub_2247850((__int64)&v70, (__int64)&v68) )
        goto LABEL_4;
      v16 = *(_BYTE *)(v9 + 32);
      v17 = 16;
      n = 22;
      v11 = 22;
LABEL_111:
      v73[0] = (__int64)&unk_4FD67D8;
      if ( !v16 )
      {
        v59 = v17;
        v67 = 0;
        v62 = 0x7FFFFFFFFFFFFFFFLL / v17;
        v63 = *(_BYTE *)(v9 + 328);
        if ( !v63 )
        {
          v65 = 0;
          v27 = 0;
          v15 = 0;
LABEL_57:
          v36 = 1;
          goto LABEL_58;
        }
        v60 = 0x7FFFFFFFFFFFFFFFLL;
        v13 = 0;
        v15 = 0;
        goto LABEL_114;
      }
      v67 = 0;
      v13 = 0;
      v15 = 0;
      v63 = v16;
      v16 = 0;
      v57 = v11;
      v11 = v17;
LABEL_138:
      sub_2215AB0(v73, 0x20u);
      goto LABEL_39;
    }
  }
  if ( sub_2247850((__int64)&v70, (__int64)&v68) )
  {
    v16 = *(_BYTE *)(v9 + 32);
    n = v11;
    v17 = v11;
    goto LABEL_111;
  }
LABEL_4:
  v12 = sub_2247910((__int64)&v70);
  v13 = v12;
  v67 = *(_DWORD *)(v9 + 224) == v12;
  if ( *(_DWORD *)(v9 + 224) == v12 || *(_DWORD *)(v9 + 228) == v12 )
  {
    v14 = *(_BYTE *)(v9 + 32);
    if ( v14 && *(_DWORD *)(v9 + 76) == v12 || *(_DWORD *)(v9 + 72) == v12 )
      goto LABEL_17;
    sub_2240940(v70);
    LODWORD(v71) = -1;
    v63 = sub_2247850((__int64)&v70, (__int64)&v68);
    if ( v63 )
    {
      v14 = *(_BYTE *)(v9 + 32);
      v15 = 0;
      v16 = 0;
      if ( v11 != 16 )
        goto LABEL_37;
LABEL_10:
      n = 22;
      v57 = 22;
      goto LABEL_38;
    }
    v13 = sub_2247910((__int64)&v70);
  }
  v14 = *(_BYTE *)(v9 + 32);
LABEL_17:
  v15 = 0;
  v16 = 0;
  while ( 1 )
  {
    if ( v14 && *(_DWORD *)(v9 + 76) == v13 || *(_DWORD *)(v9 + 72) == v13 )
    {
LABEL_74:
      v63 = 0;
      goto LABEL_36;
    }
    if ( *(_DWORD *)(v9 + 240) != v13 )
      break;
    v40 = v16 ^ 1 | (v11 == 10);
    if ( !v40 )
      goto LABEL_23;
    v16 = v11 == 8 || v10 == 0;
    if ( !v16 )
    {
      ++v15;
      v16 = v40;
      goto LABEL_27;
    }
    v18 = v70;
    v15 = 0;
    v11 = 8;
    v19 = v70[2];
    if ( v19 < v70[3] )
    {
LABEL_28:
      LODWORD(v71) = -1;
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
        v70 = 0;
        v22 = 1;
      }
      goto LABEL_33;
    }
LABEL_69:
    (*(void (__fastcall **)(_QWORD *))(*v18 + 80LL))(v18);
    v18 = v70;
    LODWORD(v71) = -1;
    if ( v70 )
      goto LABEL_29;
    v22 = 1;
LABEL_33:
    v23 = (_DWORD)v69 == -1;
    v24 = v23 & (v68 != 0);
    if ( v24 )
    {
      v42 = (int *)v68[2];
      if ( (unsigned __int64)v42 >= v68[3] )
      {
        v66 = v24;
        v64 = v22;
        v55 = (*(__int64 (**)(void))(*v68 + 72LL))();
        v24 = v66;
        v22 = v64;
        v43 = v55;
      }
      else
      {
        v43 = *v42;
      }
      v23 = 0;
      if ( v43 == -1 )
      {
        v68 = 0;
        v23 = v24;
      }
    }
    if ( v22 == v23 )
    {
      v63 = 1;
      v14 = *(_BYTE *)(v9 + 32);
      goto LABEL_36;
    }
    v13 = v71;
    if ( v70 && (_DWORD)v71 == -1 )
    {
      v53 = (_DWORD *)v70[2];
      v13 = (unsigned __int64)v53 >= v70[3] ? (*(__int64 (**)(void))(*v70 + 72LL))() : *v53;
      if ( v13 == -1 )
        v70 = 0;
    }
    v14 = *(_BYTE *)(v9 + 32);
    if ( !v16 )
      goto LABEL_74;
  }
  if ( !v16 )
    goto LABEL_74;
LABEL_23:
  if ( *(_DWORD *)(v9 + 232) == v13 || *(_DWORD *)(v9 + 236) == v13 )
  {
    if ( v10 != 0 && v11 != 16 )
    {
      v63 = 0;
      v16 = 1;
      n = v11;
      v57 = v11;
      goto LABEL_38;
    }
    v15 = 0;
    v16 = 0;
    v11 = 16;
LABEL_27:
    v18 = v70;
    v19 = v70[2];
    if ( v19 < v70[3] )
      goto LABEL_28;
    goto LABEL_69;
  }
  v63 = 0;
  v16 = 1;
LABEL_36:
  if ( v11 == 16 )
    goto LABEL_10;
LABEL_37:
  v57 = v11;
  n = v11;
LABEL_38:
  v73[0] = (__int64)&unk_4FD67D8;
  if ( v14 )
    goto LABEL_138;
LABEL_39:
  v25 = 0x7FFFFFFFFFFFFFFFLL;
  if ( v67 )
    v25 = 0x8000000000000000LL;
  v59 = v11;
  v60 = v25;
  v26 = v25 % v11;
  v58 = v25 / v11;
  v62 = v58;
  v65 = *(_BYTE *)(v9 + 328);
  if ( v65 )
  {
LABEL_114:
    if ( v63 )
    {
      v36 = v63;
      v65 = 0;
      v27 = 0;
      v63 = 0;
    }
    else
    {
      v65 = 0;
      v27 = 0;
      v48 = *(_BYTE *)(v9 + 32);
      if ( !v48 || *(_DWORD *)(v9 + 76) != v13 )
        goto LABEL_117;
LABEL_128:
      if ( v15 )
      {
        sub_2215DF0(v73, v15);
        v51 = v70;
        v15 = 0;
        v52 = v70[2];
        if ( v52 >= v70[3] )
        {
LABEL_130:
          (*(void (__fastcall **)(_QWORD *))(*v51 + 80LL))(v51);
          goto LABEL_125;
        }
        while ( 1 )
        {
          v51[2] = v52 + 4;
LABEL_125:
          LODWORD(v71) = -1;
          v36 = sub_2247850((__int64)&v70, (__int64)&v68);
          if ( v36 )
            break;
          v13 = sub_2247910((__int64)&v70);
          v48 = *(_BYTE *)(v9 + 32);
          if ( v48 && *(_DWORD *)(v9 + 76) == v13 )
            goto LABEL_128;
LABEL_117:
          if ( v13 == *(_DWORD *)(v9 + 72) || (v49 = wmemchr((const wchar_t *)(v9 + 240), v13, n)) == 0 )
          {
LABEL_109:
            v36 = 0;
            goto LABEL_58;
          }
          v50 = ((__int64)v49 - v9 - 240) >> 2;
          if ( (int)v50 > 15 )
            LODWORD(v50) = v50 - 6;
          if ( v62 < v27 )
          {
            v65 = 1;
          }
          else
          {
            v41 = v60 - (int)v50 < v27 * v59;
            v27 = (int)v50 + v27 * v59;
            v65 |= v41;
            ++v15;
          }
          v51 = v70;
          v52 = v70[2];
          if ( v52 >= v70[3] )
            goto LABEL_130;
        }
      }
      else
      {
        v63 = v48;
        v36 = 0;
      }
    }
    goto LABEL_58;
  }
  if ( !v63 )
  {
    v27 = 0;
    while ( v57 > 10 )
    {
      if ( (unsigned int)(v13 - 48) > 9 )
      {
        if ( (unsigned int)(v13 - 97) > 5 )
        {
          if ( (unsigned int)(v13 - 65) > 5 )
            goto LABEL_109;
          v28 = v13 - 55;
        }
        else
        {
          v28 = v13 - 87;
        }
        goto LABEL_48;
      }
      v28 = v13 - 48;
      if ( v58 < v27 )
      {
LABEL_49:
        v29 = v70;
        v65 = 1;
        v30 = v70[2];
        if ( v30 < v70[3] )
          goto LABEL_50;
        goto LABEL_78;
      }
LABEL_77:
      v29 = v70;
      v26 = v60 - v28;
      v41 = v26 < v27 * v59;
      v27 = v28 + v27 * v59;
      v30 = v70[2];
      LOBYTE(v26) = v41;
      ++v15;
      v65 |= v41;
      if ( v30 < v70[3] )
      {
LABEL_50:
        LODWORD(v71) = -1;
        v29[2] = v30 + 4;
        goto LABEL_51;
      }
LABEL_78:
      (*(void (__fastcall **)(_QWORD *, __int64 *, unsigned __int64))(*v29 + 80LL))(v29, v73, v26);
      v29 = v70;
      LODWORD(v71) = -1;
      if ( !v70 )
      {
        v33 = 1;
        goto LABEL_55;
      }
LABEL_51:
      v31 = (int *)v29[2];
      if ( (unsigned __int64)v31 >= v29[3] )
        v32 = (*(__int64 (__fastcall **)(_QWORD *, __int64 *, unsigned __int64))(*v29 + 72LL))(v29, v73, v26);
      else
        v32 = *v31;
      v33 = 0;
      if ( v32 == -1 )
      {
        v70 = 0;
        v33 = 1;
      }
LABEL_55:
      v34 = (_DWORD)v69 == -1;
      v35 = v34 & (v68 != 0);
      if ( v35 )
      {
        v47 = (_DWORD *)v68[2];
        v26 = (unsigned __int64)v47 >= v68[3]
            ? (*(unsigned int (__fastcall **)(_QWORD *, __int64 *, unsigned __int64))(*v68 + 72LL))(v68, v73, v26)
            : (unsigned int)*v47;
        v34 = 0;
        if ( (_DWORD)v26 == -1 )
        {
          v68 = 0;
          v34 = v35;
        }
      }
      if ( v34 == v33 )
        goto LABEL_57;
      v13 = v71;
      if ( (_DWORD)v71 == -1 && v70 )
      {
        v44 = (wchar_t *)v70[2];
        if ( (unsigned __int64)v44 >= v70[3] )
        {
          v45 = (*(__int64 (__fastcall **)(_QWORD *, __int64 *, unsigned __int64))(*v70 + 72LL))(v70, v73, v26);
          v13 = v45;
        }
        else
        {
          v13 = *v44;
          v45 = *v44;
        }
        if ( v45 == -1 )
          v70 = 0;
      }
    }
    if ( v13 <= 47 || (int)n + 48 <= v13 )
      goto LABEL_109;
    v28 = v13 - 48;
LABEL_48:
    if ( v58 < v27 )
      goto LABEL_49;
    goto LABEL_77;
  }
  v36 = v63;
  v27 = 0;
  v63 = 0;
LABEL_58:
  v37 = v73[0];
  if ( *(_QWORD *)(v73[0] - 24) )
  {
    sub_2215DF0(v73, v15);
    if ( !(unsigned __int8)sub_2255280(*(_QWORD *)(v9 + 16), *(_QWORD *)(v9 + 24), v73) )
      *a7 = 4;
    v37 = v73[0];
    if ( !v15 && v16 != 1 && !*(_QWORD *)(v73[0] - 24) )
      goto LABEL_61;
  }
  else if ( v16 != 1 && !v15 )
  {
    goto LABEL_61;
  }
  if ( v63 )
  {
LABEL_61:
    *a8 = 0;
    *a7 = 4;
    goto LABEL_62;
  }
  if ( v65 )
  {
    v46 = 0x8000000000000000LL;
    if ( !v67 )
      v46 = 0x7FFFFFFFFFFFFFFFLL;
    *a8 = v46;
    *a7 = 4;
  }
  else
  {
    v54 = -(__int64)v27;
    if ( !v67 )
      v54 = v27;
    *a8 = v54;
  }
LABEL_62:
  if ( v36 )
    *a7 |= 2u;
  v38 = v70;
  if ( (_UNKNOWN *)(v37 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v56 = _InterlockedExchangeAdd((volatile signed __int32 *)(v37 - 8), 0xFFFFFFFF);
    }
    else
    {
      v56 = *(_DWORD *)(v37 - 8);
      *(_DWORD *)(v37 - 8) = v56 - 1;
    }
    if ( v56 <= 0 )
      j_j___libc_free_0_1(v37 - 24);
  }
  return v38;
}
