// Function: sub_224EB20
// Address: 0x224eb20
//
_QWORD *__fastcall sub_224EB20(
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
  bool v12; // r14
  wchar_t v13; // eax
  wchar_t v14; // ebp
  char v15; // cl
  bool v16; // r12
  int v17; // r14d
  _QWORD *v18; // rdi
  unsigned __int64 v19; // rax
  int *v20; // rax
  int v21; // eax
  char v22; // cl
  char v23; // al
  char v24; // dl
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // r15
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
  char v46; // al
  wchar_t *v47; // rax
  __int64 v48; // rax
  _QWORD *v49; // rdi
  unsigned __int64 v50; // rax
  _DWORD *v51; // rax
  int v52; // eax
  int v53; // eax
  unsigned __int64 v54; // [rsp+0h] [rbp-A8h]
  int v55; // [rsp+8h] [rbp-A0h]
  __int64 v56; // [rsp+20h] [rbp-88h]
  size_t n; // [rsp+30h] [rbp-78h]
  int v58; // [rsp+38h] [rbp-70h]
  char v59; // [rsp+38h] [rbp-70h]
  char v60; // [rsp+3Dh] [rbp-6Bh]
  char v61; // [rsp+3Dh] [rbp-6Bh]
  char v62; // [rsp+3Eh] [rbp-6Ah]
  bool v63; // [rsp+3Fh] [rbp-69h]
  _QWORD *v64; // [rsp+40h] [rbp-68h] BYREF
  __int64 v65; // [rsp+48h] [rbp-60h]
  _QWORD *v66; // [rsp+50h] [rbp-58h] BYREF
  __int64 v67; // [rsp+58h] [rbp-50h]
  char v68; // [rsp+66h] [rbp-42h] BYREF
  __int64 v69[8]; // [rsp+68h] [rbp-40h] BYREF

  v66 = a2;
  v67 = a3;
  v64 = a4;
  v65 = a5;
  v9 = sub_22462F0((__int64)&v68, (__int64 *)(a6 + 208));
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
      v12 = sub_2247850((__int64)&v66, (__int64)&v64);
      if ( v12 )
      {
        v63 = 0;
        v15 = *(_BYTE *)(v9 + 32);
        v14 = 0;
        v16 = 0;
        n = 22;
        v58 = 0;
        v55 = 22;
        goto LABEL_38;
      }
      goto LABEL_4;
    }
  }
  v12 = sub_2247850((__int64)&v66, (__int64)&v64);
  if ( v12 )
  {
    v55 = v11;
    v15 = *(_BYTE *)(v9 + 32);
    v14 = 0;
    n = v11;
    v16 = 0;
    v58 = 0;
    v63 = 0;
    goto LABEL_38;
  }
LABEL_4:
  v13 = sub_2247910((__int64)&v66);
  v14 = v13;
  v63 = *(_DWORD *)(v9 + 224) == v13;
  if ( *(_DWORD *)(v9 + 224) == v13 || *(_DWORD *)(v9 + 228) == v13 )
  {
    v15 = *(_BYTE *)(v9 + 32);
    if ( v15 && *(_DWORD *)(v9 + 76) == v13 || *(_DWORD *)(v9 + 72) == v13 )
      goto LABEL_17;
    sub_2240940(v66);
    LODWORD(v67) = -1;
    v12 = sub_2247850((__int64)&v66, (__int64)&v64);
    if ( v12 )
    {
      v58 = 0;
      v15 = *(_BYTE *)(v9 + 32);
      v16 = 0;
      if ( v11 != 16 )
        goto LABEL_37;
LABEL_10:
      n = 22;
      v55 = 22;
      goto LABEL_38;
    }
    v14 = sub_2247910((__int64)&v66);
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
      v58 = v17;
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
    v18 = v66;
    v17 = 0;
    v11 = 8;
    v19 = v66[2];
    if ( v19 < v66[3] )
    {
LABEL_28:
      LODWORD(v67) = -1;
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
        v66 = 0;
        v22 = 1;
      }
      goto LABEL_33;
    }
LABEL_68:
    (*(void (__fastcall **)(_QWORD *))(*v18 + 80LL))(v18);
    v18 = v66;
    LODWORD(v67) = -1;
    if ( v66 )
      goto LABEL_29;
    v22 = 1;
LABEL_33:
    v23 = (_DWORD)v65 == -1;
    v24 = v23 & (v64 != 0);
    if ( v24 )
    {
      v41 = (int *)v64[2];
      if ( (unsigned __int64)v41 >= v64[3] )
      {
        v61 = v24;
        v59 = v22;
        v52 = (*(__int64 (**)(void))(*v64 + 72LL))();
        v24 = v61;
        v22 = v59;
        v42 = v52;
      }
      else
      {
        v42 = *v41;
      }
      v23 = 0;
      if ( v42 == -1 )
      {
        v64 = 0;
        v23 = v24;
      }
    }
    if ( v22 == v23 )
    {
      v15 = *(_BYTE *)(v9 + 32);
      v58 = v17;
      v12 = 1;
      goto LABEL_36;
    }
    v14 = v67;
    if ( v66 && (_DWORD)v67 == -1 )
    {
      v51 = (_DWORD *)v66[2];
      v14 = (unsigned __int64)v51 >= v66[3] ? (*(__int64 (**)(void))(*v66 + 72LL))() : *v51;
      if ( v14 == -1 )
        v66 = 0;
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
      v58 = v17;
      v12 = 0;
      v16 = 1;
      n = v11;
      v55 = v11;
      goto LABEL_38;
    }
    v17 = 0;
    v16 = 0;
    v11 = 16;
LABEL_27:
    v18 = v66;
    v19 = v66[2];
    if ( v19 < v66[3] )
      goto LABEL_28;
    goto LABEL_68;
  }
  v58 = v17;
  v16 = 1;
  v12 = 0;
LABEL_36:
  if ( v11 == 16 )
    goto LABEL_10;
LABEL_37:
  v55 = v11;
  n = v11;
LABEL_38:
  v69[0] = (__int64)&unk_4FD67D8;
  if ( v15 )
    sub_2215AB0(v69, 0x20u);
  v25 = 0xFFFFFFFFFFFFFFFFLL % v11;
  v56 = v11;
  v54 = 0xFFFFFFFFFFFFFFFFLL / v11;
  v62 = *(_BYTE *)(v9 + 328);
  if ( v62 )
  {
    if ( v12 )
    {
      v35 = v12;
      v60 = 0;
      v26 = 0;
      v12 = 0;
    }
    else
    {
      v60 = 0;
      v26 = 0;
      v46 = *(_BYTE *)(v9 + 32);
      if ( !v46 || *(_DWORD *)(v9 + 76) != v14 )
        goto LABEL_110;
LABEL_121:
      if ( v58 )
      {
        sub_2215DF0(v69, v58);
        v49 = v66;
        v58 = 0;
        v50 = v66[2];
        if ( v50 >= v66[3] )
        {
LABEL_123:
          (*(void (__fastcall **)(_QWORD *))(*v49 + 80LL))(v49);
          goto LABEL_118;
        }
        while ( 1 )
        {
          v49[2] = v50 + 4;
LABEL_118:
          LODWORD(v67) = -1;
          v35 = sub_2247850((__int64)&v66, (__int64)&v64);
          if ( v35 )
            break;
          v14 = sub_2247910((__int64)&v66);
          v46 = *(_BYTE *)(v9 + 32);
          if ( v46 && *(_DWORD *)(v9 + 76) == v14 )
            goto LABEL_121;
LABEL_110:
          if ( v14 == *(_DWORD *)(v9 + 72) || (v47 = wmemchr((const wchar_t *)(v9 + 240), v14, n)) == 0 )
          {
LABEL_106:
            v35 = 0;
            break;
          }
          v48 = ((__int64)v47 - v9 - 240) >> 2;
          if ( (int)v48 > 15 )
            LODWORD(v48) = v48 - 6;
          if ( v54 < v26 )
          {
            v60 = v62;
          }
          else
          {
            v40 = ~(__int64)(int)v48 < v26 * v11;
            v26 = (int)v48 + v26 * v11;
            ++v58;
            v60 |= v40;
          }
          v49 = v66;
          v50 = v66[2];
          if ( v50 >= v66[3] )
            goto LABEL_123;
        }
      }
      else
      {
        v35 = 0;
        v12 = v46;
      }
    }
  }
  else
  {
    v60 = 0;
    if ( !v12 )
    {
      v26 = 0;
      while ( v55 > 10 )
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
        if ( v54 < v26 )
        {
LABEL_48:
          v28 = v66;
          v60 = 1;
          v29 = v66[2];
          if ( v29 < v66[3] )
            goto LABEL_49;
          goto LABEL_77;
        }
LABEL_76:
        v28 = v66;
        v25 = ~(__int64)v27;
        v40 = v25 < v26 * v56;
        v26 = v27 + v26 * v56;
        v29 = v66[2];
        LOBYTE(v25) = v40;
        ++v58;
        v60 |= v40;
        if ( v29 < v66[3] )
        {
LABEL_49:
          LODWORD(v67) = -1;
          v28[2] = v29 + 4;
          goto LABEL_50;
        }
LABEL_77:
        (*(void (__fastcall **)(_QWORD *, __int64 *, unsigned __int64))(*v28 + 80LL))(v28, v69, v25);
        v28 = v66;
        LODWORD(v67) = -1;
        if ( !v66 )
        {
          v32 = 1;
          goto LABEL_54;
        }
LABEL_50:
        v30 = (int *)v28[2];
        if ( (unsigned __int64)v30 >= v28[3] )
          v31 = (*(__int64 (__fastcall **)(_QWORD *, __int64 *, unsigned __int64))(*v28 + 72LL))(v28, v69, v25);
        else
          v31 = *v30;
        v32 = 0;
        if ( v31 == -1 )
        {
          v66 = 0;
          v32 = 1;
        }
LABEL_54:
        v33 = (_DWORD)v65 == -1;
        v34 = v33 & (v64 != 0);
        if ( v34 )
        {
          v45 = (_DWORD *)v64[2];
          v25 = (unsigned __int64)v45 >= v64[3]
              ? (*(unsigned int (__fastcall **)(_QWORD *, __int64 *, unsigned __int64))(*v64 + 72LL))(v64, v69, v25)
              : (unsigned int)*v45;
          v33 = 0;
          if ( (_DWORD)v25 == -1 )
          {
            v64 = 0;
            v33 = v34;
          }
        }
        if ( v32 == v33 )
        {
          v35 = 1;
          goto LABEL_57;
        }
        v14 = v67;
        if ( (_DWORD)v67 == -1 && v66 )
        {
          v43 = (wchar_t *)v66[2];
          if ( (unsigned __int64)v43 >= v66[3] )
          {
            v44 = (*(__int64 (__fastcall **)(_QWORD *, __int64 *, unsigned __int64))(*v66 + 72LL))(v66, v69, v25);
            v14 = v44;
          }
          else
          {
            v14 = *v43;
            v44 = *v43;
          }
          if ( v44 == -1 )
            v66 = 0;
        }
      }
      if ( v14 <= 47 || (int)n + 48 <= v14 )
        goto LABEL_106;
      v27 = v14 - 48;
LABEL_47:
      if ( v54 < v26 )
        goto LABEL_48;
      goto LABEL_76;
    }
    v35 = v12;
    v26 = 0;
    v12 = 0;
  }
LABEL_57:
  v36 = v69[0];
  if ( *(_QWORD *)(v69[0] - 24) )
  {
    sub_2215DF0(v69, v58);
    if ( !(unsigned __int8)sub_2255280(*(_QWORD *)(v9 + 16), *(_QWORD *)(v9 + 24), v69) )
      *a7 = 4;
    v36 = v69[0];
    if ( !v58 && !v16 && !*(_QWORD *)(v69[0] - 24) )
      goto LABEL_60;
  }
  else if ( !v58 && !v16 )
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
  if ( v60 )
  {
    *a8 = -1;
    *a7 = 4;
  }
  else
  {
    if ( v63 )
      v26 = -(__int64)v26;
    *a8 = v26;
  }
LABEL_61:
  if ( v35 )
    *a7 |= 2u;
  v37 = v66;
  if ( (_UNKNOWN *)(v36 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v53 = _InterlockedExchangeAdd((volatile signed __int32 *)(v36 - 8), 0xFFFFFFFF);
    }
    else
    {
      v53 = *(_DWORD *)(v36 - 8);
      *(_DWORD *)(v36 - 8) = v53 - 1;
    }
    if ( v53 <= 0 )
      j_j___libc_free_0_1(v36 - 24);
  }
  return v37;
}
