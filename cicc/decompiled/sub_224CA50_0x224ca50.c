// Function: sub_224CA50
// Address: 0x224ca50
//
_QWORD *__fastcall sub_224CA50(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        int *a8)
{
  __int64 v9; // rbx
  int v10; // r14d
  int v11; // r12d
  bool v12; // r15
  wchar_t v13; // eax
  wchar_t v14; // r13d
  char v15; // cl
  bool v16; // bp
  int v17; // r15d
  _QWORD *v18; // rdi
  unsigned __int64 v19; // rax
  int *v20; // rax
  int v21; // eax
  char v22; // cl
  char v23; // al
  char v24; // dl
  char v25; // r14
  __int64 v26; // rdx
  int v27; // r13d
  _QWORD *v28; // rdi
  unsigned __int64 v29; // rax
  int *v30; // rax
  int v31; // eax
  char v32; // r14
  char v33; // al
  char v34; // r13
  __int64 v35; // rdx
  _QWORD *v36; // r12
  bool v38; // al
  int *v39; // rax
  int v40; // esi
  wchar_t *v41; // rax
  wchar_t v42; // eax
  _DWORD *v43; // rax
  char v44; // al
  wchar_t *v45; // rax
  __int64 v46; // rax
  _QWORD *v47; // rdi
  unsigned __int64 v48; // rax
  int *v49; // rax
  int v50; // eax
  char v51; // al
  char v52; // r13
  _DWORD *v53; // rax
  _DWORD *v54; // rax
  int v55; // edx
  int v56; // eax
  int v57; // eax
  int v58; // eax
  int v59; // [rsp+4h] [rbp-A4h]
  char v60; // [rsp+4h] [rbp-A4h]
  unsigned int v61; // [rsp+8h] [rbp-A0h]
  char v62; // [rsp+8h] [rbp-A0h]
  unsigned int v63; // [rsp+Ch] [rbp-9Ch]
  size_t n; // [rsp+10h] [rbp-98h]
  wchar_t na; // [rsp+10h] [rbp-98h]
  wchar_t *s; // [rsp+28h] [rbp-80h]
  int v67; // [rsp+38h] [rbp-70h]
  char v68; // [rsp+38h] [rbp-70h]
  char v69; // [rsp+3Eh] [rbp-6Ah]
  bool v70; // [rsp+3Fh] [rbp-69h]
  _QWORD *v71; // [rsp+40h] [rbp-68h] BYREF
  __int64 v72; // [rsp+48h] [rbp-60h]
  _QWORD *v73; // [rsp+50h] [rbp-58h] BYREF
  __int64 v74; // [rsp+58h] [rbp-50h]
  char v75; // [rsp+66h] [rbp-42h] BYREF
  __int64 v76[8]; // [rsp+68h] [rbp-40h] BYREF

  v73 = a2;
  v74 = a3;
  v71 = a4;
  v72 = a5;
  v9 = sub_22462F0((__int64)&v75, (__int64 *)(a6 + 208));
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
      v12 = sub_2247850((__int64)&v73, (__int64)&v71);
      if ( v12 )
      {
        v70 = 0;
        v15 = *(_BYTE *)(v9 + 32);
        v14 = 0;
        v16 = 0;
        n = 22;
        v59 = 0;
        v67 = 22;
        goto LABEL_38;
      }
      goto LABEL_4;
    }
  }
  v12 = sub_2247850((__int64)&v73, (__int64)&v71);
  if ( v12 )
  {
    v67 = v11;
    v15 = *(_BYTE *)(v9 + 32);
    v14 = 0;
    n = v11;
    v16 = 0;
    v59 = 0;
    v70 = 0;
    goto LABEL_38;
  }
LABEL_4:
  v13 = sub_2247910((__int64)&v73);
  v14 = v13;
  v70 = *(_DWORD *)(v9 + 224) == v13;
  if ( *(_DWORD *)(v9 + 224) == v13 || *(_DWORD *)(v9 + 228) == v13 )
  {
    v15 = *(_BYTE *)(v9 + 32);
    if ( v15 && *(_DWORD *)(v9 + 76) == v13 || *(_DWORD *)(v9 + 72) == v13 )
      goto LABEL_17;
    sub_2240940(v73);
    LODWORD(v74) = -1;
    v12 = sub_2247850((__int64)&v73, (__int64)&v71);
    if ( v12 )
    {
      v59 = 0;
      v15 = *(_BYTE *)(v9 + 32);
      v16 = 0;
      if ( v11 != 16 )
        goto LABEL_37;
LABEL_10:
      n = 22;
      v67 = 22;
      goto LABEL_38;
    }
    v14 = sub_2247910((__int64)&v73);
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
    v38 = !v16 || v11 == 10;
    if ( !v38 )
      goto LABEL_23;
    v16 = v11 == 8 || v10 == 0;
    if ( !v16 )
    {
      ++v17;
      v16 = v38;
      goto LABEL_27;
    }
    v18 = v73;
    v17 = 0;
    v11 = 8;
    v19 = v73[2];
    if ( v19 < v73[3] )
    {
LABEL_28:
      LODWORD(v74) = -1;
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
        v73 = 0;
        v22 = 1;
      }
      goto LABEL_33;
    }
LABEL_68:
    (*(void (__fastcall **)(_QWORD *))(*v18 + 80LL))(v18);
    v18 = v73;
    LODWORD(v74) = -1;
    if ( v73 )
      goto LABEL_29;
    v22 = 1;
LABEL_33:
    v23 = (_DWORD)v72 == -1;
    v24 = v23 & (v71 != 0);
    if ( v24 )
    {
      v39 = (int *)v71[2];
      if ( (unsigned __int64)v39 >= v71[3] )
      {
        v62 = v24;
        v60 = v22;
        v57 = (*(__int64 (**)(void))(*v71 + 72LL))();
        v24 = v62;
        v22 = v60;
        v40 = v57;
      }
      else
      {
        v40 = *v39;
      }
      v23 = 0;
      if ( v40 == -1 )
      {
        v71 = 0;
        v23 = v24;
      }
    }
    if ( v23 == v22 )
    {
      v15 = *(_BYTE *)(v9 + 32);
      v59 = v17;
      v12 = 1;
      goto LABEL_36;
    }
    v14 = v74;
    if ( v73 && (_DWORD)v74 == -1 )
    {
      v53 = (_DWORD *)v73[2];
      v14 = (unsigned __int64)v53 >= v73[3] ? (*(__int64 (**)(void))(*v73 + 72LL))() : *v53;
      if ( v14 == -1 )
        v73 = 0;
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
      v67 = v11;
      goto LABEL_38;
    }
    v17 = 0;
    v16 = 0;
    v11 = 16;
LABEL_27:
    v18 = v73;
    v19 = v73[2];
    if ( v19 < v73[3] )
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
  v67 = v11;
  n = v11;
LABEL_38:
  v76[0] = (__int64)&unk_4FD67D8;
  if ( v15 )
    sub_2215AB0(v76, 0x20u);
  v25 = *(_BYTE *)(v9 + 328);
  v26 = 0xFFFFFFFF % v11;
  v63 = 0xFFFFFFFF / v11;
  if ( v25 )
  {
    if ( v12 )
    {
      v25 = v12;
      v61 = 0;
      v12 = 0;
      v69 = 0;
    }
    else
    {
      v69 = 0;
      s = (wchar_t *)(v9 + 240);
      v44 = *(_BYTE *)(v9 + 32);
      v61 = 0;
      if ( !v44 || *(_DWORD *)(v9 + 76) != v14 )
        goto LABEL_110;
LABEL_126:
      if ( v59 )
      {
        sub_2215DF0(v76, v59);
        v47 = v73;
        v59 = 0;
        v48 = v73[2];
        if ( v48 >= v73[3] )
        {
LABEL_128:
          (*(void (__fastcall **)(_QWORD *))(*v47 + 80LL))(v47);
          v47 = v73;
          LODWORD(v74) = -1;
          if ( !v73 )
          {
            v68 = v25;
            goto LABEL_122;
          }
          goto LABEL_118;
        }
        while ( 1 )
        {
          LODWORD(v74) = -1;
          v47[2] = v48 + 4;
LABEL_118:
          v49 = (int *)v47[2];
          if ( (unsigned __int64)v49 >= v47[3] )
            v50 = (*(__int64 (__fastcall **)(_QWORD *))(*v47 + 72LL))(v47);
          else
            v50 = *v49;
          v68 = 0;
          if ( v50 == -1 )
          {
            v73 = 0;
            v68 = v25;
          }
LABEL_122:
          v51 = (_DWORD)v72 == -1;
          v52 = v51 & (v71 != 0);
          if ( v52 )
          {
            v54 = (_DWORD *)v71[2];
            v55 = (unsigned __int64)v54 >= v71[3] ? (*(__int64 (**)(void))(*v71 + 72LL))() : *v54;
            v51 = 0;
            if ( v55 == -1 )
            {
              v71 = 0;
              v51 = v52;
            }
          }
          if ( v51 == v68 )
            break;
          v14 = sub_2247910((__int64)&v73);
          v44 = *(_BYTE *)(v9 + 32);
          if ( v44 && *(_DWORD *)(v9 + 76) == v14 )
            goto LABEL_126;
LABEL_110:
          if ( v14 == *(_DWORD *)(v9 + 72) || (v45 = wmemchr(s, v14, n)) == 0 )
          {
LABEL_106:
            v25 = 0;
            break;
          }
          v46 = v45 - s;
          if ( (int)v46 > 15 )
            LODWORD(v46) = v46 - 6;
          if ( v63 < v61 )
          {
            v69 = v25;
          }
          else
          {
            v69 |= ~(_DWORD)v46 < v11 * v61;
            ++v59;
            v61 = v11 * v61 + v46;
          }
          v47 = v73;
          v48 = v73[2];
          if ( v48 >= v73[3] )
            goto LABEL_128;
        }
      }
      else
      {
        v25 = 0;
        v12 = v44;
      }
    }
  }
  else
  {
    v69 = 0;
    if ( !v12 )
    {
      v61 = 0;
      na = n + 48;
      while ( v67 > 10 )
      {
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
        if ( v63 < v61 )
        {
LABEL_48:
          v28 = v73;
          v69 = 1;
          v29 = v73[2];
          if ( v29 < v73[3] )
            goto LABEL_49;
          goto LABEL_77;
        }
LABEL_76:
        v28 = v73;
        v26 = (unsigned int)~v27;
        LOBYTE(v26) = (unsigned int)v26 < v11 * v61;
        v69 |= v26;
        v61 = v27 + v11 * v61;
        v29 = v73[2];
        ++v59;
        if ( v29 < v73[3] )
        {
LABEL_49:
          LODWORD(v74) = -1;
          v28[2] = v29 + 4;
          goto LABEL_50;
        }
LABEL_77:
        (*(void (__fastcall **)(_QWORD *, __int64 *, __int64))(*v28 + 80LL))(v28, v76, v26);
        v28 = v73;
        LODWORD(v74) = -1;
        if ( !v73 )
        {
          v32 = 1;
          goto LABEL_54;
        }
LABEL_50:
        v30 = (int *)v28[2];
        if ( (unsigned __int64)v30 >= v28[3] )
          v31 = (*(__int64 (__fastcall **)(_QWORD *, __int64 *, __int64))(*v28 + 72LL))(v28, v76, v26);
        else
          v31 = *v30;
        v32 = 0;
        if ( v31 == -1 )
        {
          v73 = 0;
          v32 = 1;
        }
LABEL_54:
        v33 = (_DWORD)v72 == -1;
        v34 = v33 & (v71 != 0);
        if ( v34 )
        {
          v43 = (_DWORD *)v71[2];
          v26 = (unsigned __int64)v43 >= v71[3]
              ? (*(unsigned int (__fastcall **)(_QWORD *, __int64 *, __int64))(*v71 + 72LL))(v71, v76, v26)
              : (unsigned int)*v43;
          v33 = 0;
          if ( (_DWORD)v26 == -1 )
          {
            v71 = 0;
            v33 = v34;
          }
        }
        if ( v32 == v33 )
        {
          v25 = 1;
          goto LABEL_57;
        }
        v14 = v74;
        if ( v73 && (_DWORD)v74 == -1 )
        {
          v41 = (wchar_t *)v73[2];
          if ( (unsigned __int64)v41 >= v73[3] )
          {
            v42 = (*(__int64 (__fastcall **)(_QWORD *, __int64 *, __int64))(*v73 + 72LL))(v73, v76, v26);
            v14 = v42;
          }
          else
          {
            v14 = *v41;
            v42 = *v41;
          }
          if ( v42 == -1 )
            v73 = 0;
        }
      }
      if ( v14 <= 47 || na <= v14 )
        goto LABEL_106;
      v27 = v14 - 48;
LABEL_47:
      if ( v63 < v61 )
        goto LABEL_48;
      goto LABEL_76;
    }
    v25 = v12;
    v61 = 0;
    v12 = 0;
  }
LABEL_57:
  v35 = v76[0];
  if ( *(_QWORD *)(v76[0] - 24) )
  {
    sub_2215DF0(v76, v59);
    if ( !(unsigned __int8)sub_2255280(*(_QWORD *)(v9 + 16), *(_QWORD *)(v9 + 24), v76) )
      *a7 = 4;
    v35 = v76[0];
    if ( !v59 && !v16 && !*(_QWORD *)(v76[0] - 24) )
      goto LABEL_60;
  }
  else if ( !v16 && !v59 )
  {
    goto LABEL_60;
  }
  if ( v12 )
  {
LABEL_60:
    *a8 = 0;
    *a7 = 4;
    goto LABEL_61;
  }
  if ( v69 )
  {
    *a8 = -1;
    *a7 = 4;
  }
  else
  {
    v56 = -v61;
    if ( !v70 )
      v56 = v61;
    *a8 = v56;
  }
LABEL_61:
  if ( v25 )
    *a7 |= 2u;
  v36 = v73;
  if ( (_UNKNOWN *)(v35 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v58 = _InterlockedExchangeAdd((volatile signed __int32 *)(v35 - 8), 0xFFFFFFFF);
    }
    else
    {
      v58 = *(_DWORD *)(v35 - 8);
      *(_DWORD *)(v35 - 8) = v58 - 1;
    }
    if ( v58 <= 0 )
      j_j___libc_free_0_1(v35 - 24);
  }
  return v36;
}
