// Function: sub_224C000
// Address: 0x224c000
//
_QWORD *__fastcall sub_224C000(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        _WORD *a8)
{
  __int64 v9; // rbx
  int v10; // r15d
  int v11; // r12d
  bool v12; // r14
  wchar_t v13; // eax
  wchar_t v14; // r13d
  char v15; // cl
  bool v16; // bp
  int v17; // r14d
  _QWORD *v18; // rdi
  unsigned __int64 v19; // rax
  int *v20; // rax
  int v21; // eax
  char v22; // cl
  char v23; // al
  char v24; // dl
  __int64 v25; // rdx
  int v26; // r15d
  int v27; // r13d
  _QWORD *v28; // rdi
  unsigned __int64 v29; // rax
  int *v30; // rax
  int v31; // eax
  char v32; // al
  char v33; // r13
  bool v34; // r13
  __int64 v35; // rdx
  _QWORD *v36; // r12
  bool v38; // al
  int v39; // ecx
  int *v40; // rax
  int v41; // esi
  wchar_t *v42; // rax
  wchar_t v43; // eax
  _DWORD *v44; // rax
  char v45; // al
  wchar_t *v46; // rax
  __int64 v47; // rax
  int v48; // esi
  _QWORD *v49; // rdi
  unsigned __int64 v50; // rax
  _DWORD *v51; // rax
  int v52; // eax
  int v53; // eax
  int v54; // [rsp+Ch] [rbp-9Ch]
  char v55; // [rsp+Ch] [rbp-9Ch]
  char v56; // [rsp+10h] [rbp-98h]
  int s; // [rsp+18h] [rbp-90h]
  wchar_t *sa; // [rsp+18h] [rbp-90h]
  size_t n; // [rsp+30h] [rbp-78h]
  int v60; // [rsp+38h] [rbp-70h]
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
        v54 = 0;
        s = 22;
        goto LABEL_38;
      }
      goto LABEL_4;
    }
  }
  v12 = sub_2247850((__int64)&v67, (__int64)&v65);
  if ( v12 )
  {
    s = v11;
    v15 = *(_BYTE *)(v9 + 32);
    v14 = 0;
    n = v11;
    v16 = 0;
    v54 = 0;
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
      v54 = 0;
      v15 = *(_BYTE *)(v9 + 32);
      v16 = 0;
      if ( v11 != 16 )
        goto LABEL_37;
LABEL_10:
      n = 22;
      s = 22;
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
      v54 = v17;
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
      v40 = (int *)v65[2];
      if ( (unsigned __int64)v40 >= v65[3] )
      {
        v62 = v24;
        v55 = v22;
        v52 = (*(__int64 (**)(void))(*v65 + 72LL))();
        v24 = v62;
        v22 = v55;
        v41 = v52;
      }
      else
      {
        v41 = *v40;
      }
      v23 = 0;
      if ( v41 == -1 )
      {
        v65 = 0;
        v23 = v24;
      }
    }
    if ( v22 == v23 )
    {
      v15 = *(_BYTE *)(v9 + 32);
      v54 = v17;
      v12 = 1;
      goto LABEL_36;
    }
    v14 = v68;
    if ( v67 && (_DWORD)v68 == -1 )
    {
      v51 = (_DWORD *)v67[2];
      v14 = (unsigned __int64)v51 >= v67[3] ? (*(__int64 (**)(void))(*v67 + 72LL))() : *v51;
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
      v54 = v17;
      v12 = 0;
      v16 = 1;
      n = v11;
      s = v11;
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
  v54 = v17;
  v16 = 1;
  v12 = 0;
LABEL_36:
  if ( v11 == 16 )
    goto LABEL_10;
LABEL_37:
  s = v11;
  n = v11;
LABEL_38:
  v70[0] = (__int64)&unk_4FD67D8;
  if ( v15 )
    sub_2215AB0(v70, 0x20u);
  v25 = (unsigned int)(0xFFFF % v11);
  v60 = 0xFFFF / v11;
  v63 = *(_BYTE *)(v9 + 328);
  if ( v63 )
  {
    if ( v12 )
    {
      v34 = v12;
      v61 = 0;
      LOWORD(v26) = 0;
      v12 = 0;
    }
    else
    {
      v61 = 0;
      LOWORD(v26) = 0;
      sa = (wchar_t *)(v9 + 240);
      v45 = *(_BYTE *)(v9 + 32);
      if ( !v45 || *(_DWORD *)(v9 + 76) != v14 )
        goto LABEL_110;
LABEL_121:
      if ( v54 )
      {
        sub_2215DF0(v70, v54);
        v49 = v67;
        v54 = 0;
        v50 = v67[2];
        if ( v50 >= v67[3] )
        {
LABEL_123:
          (*(void (__fastcall **)(_QWORD *))(*v49 + 80LL))(v49);
          goto LABEL_118;
        }
        while ( 1 )
        {
          v49[2] = v50 + 4;
LABEL_118:
          LODWORD(v68) = -1;
          v34 = sub_2247850((__int64)&v67, (__int64)&v65);
          if ( v34 )
            break;
          v14 = sub_2247910((__int64)&v67);
          v45 = *(_BYTE *)(v9 + 32);
          if ( v45 && *(_DWORD *)(v9 + 76) == v14 )
            goto LABEL_121;
LABEL_110:
          if ( v14 == *(_DWORD *)(v9 + 72) || (v46 = wmemchr(sa, v14, n)) == 0 )
          {
LABEL_106:
            v34 = 0;
            break;
          }
          v47 = v46 - sa;
          if ( (int)v47 > 15 )
            LODWORD(v47) = v47 - 6;
          if ( (unsigned __int16)v60 < (unsigned __int16)v26 )
          {
            v61 = v63;
          }
          else
          {
            v48 = (unsigned __int16)(v11 * v26);
            LOWORD(v26) = v11 * v26 + v47;
            ++v54;
            v61 |= v48 > 0xFFFF - (int)v47;
          }
          v49 = v67;
          v50 = v67[2];
          if ( v50 >= v67[3] )
            goto LABEL_123;
        }
      }
      else
      {
        v34 = 0;
        v12 = v45;
      }
    }
  }
  else
  {
    v61 = 0;
    if ( !v12 )
    {
      v26 = 0;
      while ( s > 10 )
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
        if ( (unsigned __int16)v60 < (unsigned __int16)v26 )
        {
LABEL_48:
          v28 = v67;
          v61 = 1;
          v29 = v67[2];
          if ( v29 < v67[3] )
            goto LABEL_49;
          goto LABEL_77;
        }
LABEL_76:
        v28 = v67;
        v25 = (unsigned int)(v11 * v26);
        v39 = (unsigned __int16)(v11 * v26);
        v26 = v25 + v27;
        ++v54;
        v61 |= v39 > 0xFFFF - v27;
        v29 = v67[2];
        if ( v29 < v67[3] )
        {
LABEL_49:
          LODWORD(v68) = -1;
          v28[2] = v29 + 4;
          goto LABEL_50;
        }
LABEL_77:
        (*(void (__fastcall **)(_QWORD *, __int64 *, __int64))(*v28 + 80LL))(v28, v70, v25);
        v28 = v67;
        LODWORD(v68) = -1;
        if ( !v67 )
        {
          v56 = 1;
          goto LABEL_54;
        }
LABEL_50:
        v30 = (int *)v28[2];
        if ( (unsigned __int64)v30 >= v28[3] )
          v31 = (*(__int64 (__fastcall **)(_QWORD *, __int64 *, __int64))(*v28 + 72LL))(v28, v70, v25);
        else
          v31 = *v30;
        v56 = 0;
        if ( v31 == -1 )
        {
          v67 = 0;
          v56 = 1;
        }
LABEL_54:
        v32 = (_DWORD)v66 == -1;
        v33 = v32 & (v65 != 0);
        if ( v33 )
        {
          v44 = (_DWORD *)v65[2];
          v25 = (unsigned __int64)v44 >= v65[3]
              ? (*(unsigned int (__fastcall **)(_QWORD *, __int64 *, __int64))(*v65 + 72LL))(v65, v70, v25)
              : (unsigned int)*v44;
          v32 = 0;
          if ( (_DWORD)v25 == -1 )
          {
            v65 = 0;
            v32 = v33;
          }
        }
        if ( v56 == v32 )
        {
          v34 = 1;
          goto LABEL_57;
        }
        v14 = v68;
        if ( (_DWORD)v68 == -1 && v67 )
        {
          v42 = (wchar_t *)v67[2];
          if ( (unsigned __int64)v42 >= v67[3] )
          {
            v43 = (*(__int64 (__fastcall **)(_QWORD *, __int64 *, __int64))(*v67 + 72LL))(v67, v70, v25);
            v14 = v43;
          }
          else
          {
            v14 = *v42;
            v43 = *v42;
          }
          if ( v43 == -1 )
            v67 = 0;
        }
      }
      if ( v14 <= 47 || (int)n + 48 <= v14 )
        goto LABEL_106;
      v27 = v14 - 48;
LABEL_47:
      if ( (unsigned __int16)v60 < (unsigned __int16)v26 )
        goto LABEL_48;
      goto LABEL_76;
    }
    v34 = v12;
    LOWORD(v26) = 0;
    v12 = 0;
  }
LABEL_57:
  v35 = v70[0];
  if ( *(_QWORD *)(v70[0] - 24) )
  {
    sub_2215DF0(v70, v54);
    if ( !(unsigned __int8)sub_2255280(*(_QWORD *)(v9 + 16), *(_QWORD *)(v9 + 24), v70) )
      *a7 = 4;
    v35 = v70[0];
    if ( !v54 && !v16 && !*(_QWORD *)(v70[0] - 24) )
      goto LABEL_60;
  }
  else if ( !v54 && !v16 )
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
  if ( v61 )
  {
    *a8 = -1;
    *a7 = 4;
  }
  else
  {
    if ( v64 )
      LOWORD(v26) = -(__int16)v26;
    *a8 = v26;
  }
LABEL_61:
  if ( v34 )
    *a7 |= 2u;
  v36 = v67;
  if ( (_UNKNOWN *)(v35 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v53 = _InterlockedExchangeAdd((volatile signed __int32 *)(v35 - 8), 0xFFFFFFFF);
    }
    else
    {
      v53 = *(_DWORD *)(v35 - 8);
      *(_DWORD *)(v35 - 8) = v53 - 1;
    }
    if ( v53 <= 0 )
      j_j___libc_free_0_1(v35 - 24);
  }
  return v36;
}
