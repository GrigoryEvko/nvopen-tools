// Function: sub_2236190
// Address: 0x2236190
//
_QWORD *__fastcall sub_2236190(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        __int64 *a8)
{
  _QWORD **v8; // rsi
  __int64 v9; // r15
  bool v10; // bp
  char v11; // al
  char v12; // r12
  char v13; // r13
  char v14; // bp
  __int64 v15; // r14
  __int64 v16; // rax
  _QWORD *v17; // rdi
  unsigned __int64 v18; // rax
  char v19; // r12
  char v20; // al
  char v21; // bp
  char *v22; // rax
  unsigned __int64 v23; // rax
  char v24; // al
  char v25; // si
  char v26; // al
  void *v27; // rax
  _QWORD *v28; // rdi
  unsigned __int64 v29; // rax
  char v30; // r12
  char v31; // al
  __int64 v32; // rax
  __int64 v33; // rbx
  _QWORD *result; // rax
  bool v35; // zf
  __int64 v36; // rax
  unsigned __int64 v37; // rax
  char v38; // dl
  int v39; // r8d
  int v40; // r8d
  char *v41; // rax
  char *v42; // rax
  unsigned int v43; // eax
  char v44; // cl
  __int64 v45; // rdx
  int v46; // r14d
  _QWORD *v47; // rdi
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rax
  char v50; // cl
  char v51; // r13
  bool v52; // al
  _BYTE *v53; // rax
  __int64 v54; // rbp
  __int64 v55; // rax
  int v56; // eax
  char *v57; // rax
  int v58; // ecx
  int v59; // eax
  int v60; // eax
  int v61; // eax
  int v62; // eax
  int v63; // edx
  bool v64; // [rsp+0h] [rbp-A8h]
  _QWORD *v65; // [rsp+0h] [rbp-A8h]
  char v66; // [rsp+20h] [rbp-88h]
  char v67; // [rsp+20h] [rbp-88h]
  int v68; // [rsp+28h] [rbp-80h]
  unsigned __int8 v69; // [rsp+28h] [rbp-80h]
  __int16 v70; // [rsp+2Dh] [rbp-7Bh]
  char v71; // [rsp+2Fh] [rbp-79h]
  _QWORD *v72; // [rsp+40h] [rbp-68h] BYREF
  __int64 v73; // [rsp+48h] [rbp-60h]
  _QWORD *v74; // [rsp+50h] [rbp-58h] BYREF
  __int64 v75; // [rsp+58h] [rbp-50h]
  char v76; // [rsp+66h] [rbp-42h] BYREF
  __int64 v77[8]; // [rsp+68h] [rbp-40h] BYREF

  v74 = a2;
  v75 = a3;
  v72 = a4;
  v73 = a5;
  v8 = &v72;
  v9 = sub_2232A70((__int64)&v76, (__int64 *)(a6 + 208));
  v10 = sub_2233E50((__int64)&v74, (__int64)&v72);
  if ( v10 )
  {
    v11 = *(_BYTE *)(v9 + 32);
    v12 = 0;
    v13 = 0;
    v68 = 0;
    goto LABEL_3;
  }
  v43 = sub_2233F00((__int64)&v74);
  v44 = *(_BYTE *)(v9 + 111);
  v45 = v43;
  v12 = v43;
  if ( v44 == (_BYTE)v43 || *(_BYTE *)(v9 + 110) == (_BYTE)v43 )
  {
    v11 = *(_BYTE *)(v9 + 32);
    if ( v11 && *(_BYTE *)(v9 + 73) == (_BYTE)v45 || *(_BYTE *)(v9 + 72) == (_BYTE)v45 )
      goto LABEL_100;
    sub_2215DF0(a8, 2 * (v44 != (char)v45) + 43);
    sub_22408B0(v74);
    v8 = &v72;
    LODWORD(v75) = -1;
    v52 = sub_2233E50((__int64)&v74, (__int64)&v72);
    if ( v52 )
    {
      v68 = 0;
      v13 = 0;
      v10 = v52;
      v11 = *(_BYTE *)(v9 + 32);
      goto LABEL_3;
    }
    v12 = sub_2233F00((__int64)&v74);
  }
  v11 = *(_BYTE *)(v9 + 32);
LABEL_100:
  v64 = v10;
  v13 = 0;
  v46 = 0;
  while ( 1 )
  {
    if ( v11 && *(_BYTE *)(v9 + 73) == v12 )
    {
      v10 = v64;
      v68 = v46;
      v77[0] = (__int64)&unk_4FD67D8;
      goto LABEL_96;
    }
    if ( *(_BYTE *)(v9 + 72) == v12 || *(_BYTE *)(v9 + 114) != v12 )
      break;
    if ( !v13 )
    {
      v54 = *(_QWORD *)(*a8 - 24);
      if ( (unsigned __int64)(v54 + 1) > *(_QWORD *)(*a8 - 16) || *(int *)(*a8 - 8) > 0 )
      {
        v8 = (_QWORD **)(v54 + 1);
        sub_2215AB0(a8, v54 + 1);
      }
      *(_BYTE *)(*a8 + *(_QWORD *)(*a8 - 24)) = 48;
      v55 = *a8;
      if ( (_UNKNOWN *)(*a8 - 24) != &unk_4FD67C0 )
      {
        *(_DWORD *)(v55 - 8) = 0;
        *(_QWORD *)(v55 - 24) = v54 + 1;
        *(_BYTE *)(v55 + v54 + 1) = 0;
      }
    }
    v47 = v74;
    ++v46;
    v48 = v74[2];
    if ( v48 >= v74[3] )
    {
      v49 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD **, __int64))(*v74 + 80LL))(v74, v8, v45);
      v47 = v74;
      LODWORD(v75) = -1;
      if ( !v74 )
      {
        v50 = 1;
        goto LABEL_109;
      }
    }
    else
    {
      LODWORD(v75) = -1;
      v74[2] = v48 + 1;
    }
    v49 = v47[3];
    v50 = 0;
    if ( v47[2] >= v49 )
    {
      LODWORD(v49) = (*(__int64 (__fastcall **)(_QWORD *, _QWORD **, __int64))(*v47 + 72LL))(v47, v8, v45);
      v50 = 0;
      if ( (_DWORD)v49 == -1 )
      {
        v74 = 0;
        v50 = 1;
      }
    }
LABEL_109:
    v51 = (_DWORD)v73 == -1;
    LOBYTE(v49) = v51 & (v72 != 0);
    v45 = (unsigned int)v49;
    if ( (_BYTE)v49 )
    {
      v51 = 0;
      if ( v72[2] >= v72[3] )
      {
        v67 = v50;
        v69 = v49;
        v56 = (*(__int64 (**)(void))(*v72 + 72LL))();
        v45 = v69;
        v50 = v67;
        if ( v56 == -1 )
        {
          v72 = 0;
          v51 = v69;
        }
      }
    }
    if ( v50 == v51 )
    {
      v68 = v46;
      v11 = *(_BYTE *)(v9 + 32);
      v13 = 1;
      v10 = 1;
      goto LABEL_3;
    }
    v12 = v75;
    if ( (_DWORD)v75 == -1 && v74 )
    {
      v57 = (char *)v74[2];
      if ( (unsigned __int64)v57 >= v74[3] )
      {
        v60 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD **, __int64))(*v74 + 72LL))(v74, v8, v45);
        v12 = v60;
        if ( v60 == -1 )
          v74 = 0;
      }
      else
      {
        v12 = *v57;
      }
    }
    v11 = *(_BYTE *)(v9 + 32);
    v13 = 1;
  }
  v68 = v46;
  v10 = v64;
LABEL_3:
  v77[0] = (__int64)&unk_4FD67D8;
  if ( v11 )
LABEL_96:
    sub_2215AB0(v77, 0x20u);
  v66 = *(_BYTE *)(v9 + 136);
  if ( !v66 )
  {
    if ( !v10 )
    {
      v70 = 0;
LABEL_7:
      v14 = v12;
      if ( (unsigned __int8)(v12 - 48) <= 9u )
      {
LABEL_8:
        v15 = *(_QWORD *)(*a8 - 24);
        if ( (unsigned __int64)(v15 + 1) > *(_QWORD *)(*a8 - 16) || *(int *)(*a8 - 8) > 0 )
          sub_2215AB0(a8, v15 + 1);
        v13 = 1;
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
        if ( *(_BYTE *)(v9 + 72) == v12 && !v70 )
        {
          sub_2215DF0(a8, 46);
          v70 = 256;
LABEL_13:
          v17 = v74;
          v18 = v74[2];
          if ( v18 >= v74[3] )
            goto LABEL_32;
LABEL_14:
          LODWORD(v75) = -1;
          v17[2] = v18 + 1;
LABEL_15:
          v19 = 0;
          if ( v17[2] >= v17[3] && (*(unsigned int (__fastcall **)(_QWORD *))(*v17 + 72LL))(v17) == -1 )
          {
            v74 = 0;
            v19 = 1;
          }
          goto LABEL_16;
        }
        if ( *(_BYTE *)(v9 + 128) != v12 && *(_BYTE *)(v9 + 134) != v12 )
          goto LABEL_46;
        v13 &= v70 ^ 1;
        if ( !v13 )
          goto LABEL_46;
        sub_2215DF0(a8, 101);
        v23 = v74[2];
        if ( v23 >= v74[3] )
          (*(void (__fastcall **)(_QWORD *))(*v74 + 80LL))(v74);
        else
          v74[2] = v23 + 1;
        LODWORD(v75) = -1;
        if ( sub_2233E50((__int64)&v74, (__int64)&v72) )
          goto LABEL_159;
        v24 = sub_2233F00((__int64)&v74);
        v12 = v24;
        if ( *(_BYTE *)(v9 + 111) == v24 )
        {
          v25 = 43;
        }
        else
        {
          v25 = 45;
          if ( *(_BYTE *)(v9 + 110) != v24 )
          {
            LOBYTE(v70) = v13;
            goto LABEL_7;
          }
        }
        sub_2215DF0(a8, v25);
        v17 = v74;
        LOBYTE(v70) = v13;
        v18 = v74[2];
        if ( v18 < v74[3] )
          goto LABEL_14;
LABEL_32:
        (*(void (__fastcall **)(_QWORD *))(*v17 + 80LL))(v17);
        v17 = v74;
        LODWORD(v75) = -1;
        if ( v74 )
          goto LABEL_15;
        v19 = 1;
LABEL_16:
        v20 = (_DWORD)v73 == -1;
        v21 = v20 & (v72 != 0);
        if ( v21 )
        {
          v20 = 0;
          if ( v72[2] >= v72[3] )
          {
            v39 = (*(__int64 (__fastcall **)(_QWORD *))(*v72 + 72LL))(v72);
            v20 = 0;
            if ( v39 == -1 )
            {
              v72 = 0;
              v20 = v21;
            }
          }
        }
        if ( v20 == v19 )
          goto LABEL_46;
        v12 = v75;
        if ( !v74 || (_DWORD)v75 != -1 )
          goto LABEL_7;
        v22 = (char *)v74[2];
        if ( (unsigned __int64)v22 >= v74[3] )
        {
          v59 = (*(__int64 (__fastcall **)(_QWORD *))(*v74 + 72LL))(v74);
          v12 = v59;
          if ( v59 == -1 )
          {
            v74 = 0;
            v12 = -1;
          }
          goto LABEL_7;
        }
        v12 = *v22;
        v14 = *v22;
        if ( (unsigned __int8)(*v22 - 48) <= 9u )
          goto LABEL_8;
      }
    }
LABEL_126:
    v33 = v77[0];
    if ( *(_QWORD *)(v77[0] - 24) )
      goto LABEL_49;
    goto LABEL_52;
  }
  if ( v10 )
    goto LABEL_126;
  v70 = 0;
  v26 = *(_BYTE *)(v9 + 32);
  while ( 2 )
  {
    while ( 2 )
    {
      if ( v26 && *(_BYTE *)(v9 + 73) == v12 )
      {
        v35 = v70 == 0;
        HIBYTE(v70) |= v70;
        if ( !v35 )
        {
          v36 = v77[0];
LABEL_155:
          v33 = v36;
          if ( *(_QWORD *)(v36 - 24) )
            goto LABEL_50;
          goto LABEL_52;
        }
        if ( !v68 )
        {
          v53 = (_BYTE *)*a8;
          if ( *(int *)(*a8 - 8) <= 0 )
          {
            if ( v53 - 24 != (_BYTE *)&unk_4FD67C0 )
            {
              *((_DWORD *)v53 - 2) = 0;
              *((_QWORD *)v53 - 3) = 0;
              *v53 = 0;
            }
          }
          else
          {
            if ( v53 - 24 != (_BYTE *)&unk_4FD67C0 )
            {
              if ( &_pthread_key_create )
              {
                v63 = _InterlockedExchangeAdd((volatile signed __int32 *)v53 - 2, 0xFFFFFFFF);
              }
              else
              {
                v63 = *((_DWORD *)v53 - 2);
                *((_DWORD *)v53 - 2) = v63 - 1;
              }
              if ( v63 <= 0 )
                j_j___libc_free_0_1((unsigned __int64)(v53 - 24));
            }
            *a8 = (__int64)&unk_4FD67D8;
          }
          goto LABEL_126;
        }
        sub_2215DF0(v77, v68);
        LOBYTE(v70) = 0;
        v68 = 0;
        goto LABEL_41;
      }
      if ( *(_BYTE *)(v9 + 72) == v12 )
      {
        v35 = v70 == 0;
        LOBYTE(v70) = HIBYTE(v70) | v70;
        v36 = v77[0];
        if ( !v35 )
          goto LABEL_155;
        if ( *(_QWORD *)(v77[0] - 24) )
          sub_2215DF0(v77, v68);
        sub_2215DF0(a8, 46);
        v28 = v74;
        HIBYTE(v70) = v66;
        v29 = v74[2];
        if ( v29 < v74[3] )
          goto LABEL_42;
      }
      else
      {
        v27 = memchr((const void *)(v9 + 114), v12, 0xAu);
        if ( v27 )
        {
          sub_2215DF0(a8, (_BYTE)v27 - (v9 + 114) + 48);
          ++v68;
          v13 = v66;
        }
        else
        {
          if ( *(_BYTE *)(v9 + 128) != v12 && *(_BYTE *)(v9 + 134) != v12 )
            goto LABEL_46;
          v32 = v77[0];
          v13 &= v70 ^ 1;
          if ( !v13 )
            goto LABEL_47;
          if ( *(_QWORD *)(v77[0] - 24) && !HIBYTE(v70) )
            sub_2215DF0(v77, v68);
          sub_2215DF0(a8, 101);
          v37 = v74[2];
          if ( v37 >= v74[3] )
            (*(void (__fastcall **)(_QWORD *))(*v74 + 80LL))(v74);
          else
            v74[2] = v37 + 1;
          LODWORD(v75) = -1;
          if ( sub_2233E50((__int64)&v74, (__int64)&v72) )
          {
LABEL_159:
            v33 = v77[0];
            if ( *(_QWORD *)(v77[0] - 24) )
              goto LABEL_50;
            goto LABEL_52;
          }
          v12 = v75;
          if ( (_DWORD)v75 == -1 && v74 )
          {
            v42 = (char *)v74[2];
            if ( (unsigned __int64)v42 >= v74[3] )
            {
              v62 = (*(__int64 (__fastcall **)(_QWORD *))(*v74 + 72LL))(v74);
              v12 = v62;
              if ( v62 == -1 )
              {
                v74 = 0;
                v12 = -1;
              }
            }
            else
            {
              v12 = *v42;
            }
          }
          v38 = *(_BYTE *)(v9 + 111);
          v26 = *(_BYTE *)(v9 + 32);
          if ( v38 != v12 && *(_BYTE *)(v9 + 110) != v12 )
            goto LABEL_77;
          if ( v26 && *(_BYTE *)(v9 + 73) == v12 )
          {
            LOBYTE(v70) = *(_BYTE *)(v9 + 32);
            v13 = v70;
            continue;
          }
          if ( *(_BYTE *)(v9 + 72) == v12 )
          {
LABEL_77:
            LOBYTE(v70) = v13;
            continue;
          }
          sub_2215DF0(a8, 2 * (v38 != v12) + 43);
          LOBYTE(v70) = v13;
        }
LABEL_41:
        v28 = v74;
        v29 = v74[2];
        if ( v29 < v74[3] )
        {
LABEL_42:
          LODWORD(v75) = -1;
          v28[2] = v29 + 1;
          goto LABEL_43;
        }
      }
      break;
    }
    (*(void (__fastcall **)(_QWORD *))(*v28 + 80LL))(v28);
    v28 = v74;
    LODWORD(v75) = -1;
    if ( v74 )
    {
LABEL_43:
      v30 = 0;
      if ( v28[2] >= v28[3] && (*(unsigned int (__fastcall **)(_QWORD *))(*v28 + 72LL))(v28) == -1 )
      {
        v74 = 0;
        v30 = v66;
      }
    }
    else
    {
      v30 = v66;
    }
    v31 = (_DWORD)v73 == -1;
    v71 = v31 & (v72 != 0);
    if ( v71 )
    {
      v31 = 0;
      if ( v72[2] >= v72[3] )
      {
        v40 = (*(__int64 (__fastcall **)(_QWORD *))(*v72 + 72LL))(v72);
        v31 = 0;
        if ( v40 == -1 )
        {
          v72 = 0;
          v31 = v71;
        }
      }
    }
    if ( v30 != v31 )
    {
      v12 = v75;
      if ( (_DWORD)v75 == -1 && v74 )
      {
        v41 = (char *)v74[2];
        if ( (unsigned __int64)v41 >= v74[3] )
        {
          v61 = (*(__int64 (__fastcall **)(_QWORD *))(*v74 + 72LL))(v74);
          v12 = v61;
          if ( v61 == -1 )
          {
            v74 = 0;
            v12 = -1;
          }
        }
        else
        {
          v12 = *v41;
        }
      }
      v26 = *(_BYTE *)(v9 + 32);
      continue;
    }
    break;
  }
LABEL_46:
  v32 = v77[0];
LABEL_47:
  v33 = v32;
  if ( !*(_QWORD *)(v32 - 24) )
    goto LABEL_52;
  if ( v70 )
    goto LABEL_50;
LABEL_49:
  sub_2215DF0(v77, v68);
  v33 = v77[0];
LABEL_50:
  if ( !(unsigned __int8)sub_2255280(*(_QWORD *)(v9 + 16), *(_QWORD *)(v9 + 24), v77) )
    *a7 = 4;
LABEL_52:
  result = v74;
  if ( (_UNKNOWN *)(v33 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v58 = _InterlockedExchangeAdd((volatile signed __int32 *)(v33 - 8), 0xFFFFFFFF);
    }
    else
    {
      v58 = *(_DWORD *)(v33 - 8);
      *(_DWORD *)(v33 - 8) = v58 - 1;
    }
    if ( v58 <= 0 )
    {
      v65 = result;
      j_j___libc_free_0_1(v33 - 24);
      return v65;
    }
  }
  return result;
}
