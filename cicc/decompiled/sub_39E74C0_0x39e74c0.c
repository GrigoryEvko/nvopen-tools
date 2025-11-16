// Function: sub_39E74C0
// Address: 0x39e74c0
//
void __fastcall sub_39E74C0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // r15
  __int64 *v6; // rbx
  __int64 v7; // rdi
  int v8; // r8d
  int v9; // r9d
  int v10; // eax
  unsigned __int64 v11; // r12
  __int64 v12; // rcx
  __int64 v13; // rax
  int v14; // r13d
  int v15; // r14d
  __int64 v16; // rbx
  __int64 v17; // r15
  _BYTE *v18; // r12
  __int64 v19; // rdx
  int v20; // eax
  __int64 v21; // rcx
  void *v22; // rdx
  int v23; // r12d
  __int64 v24; // r14
  __int64 v25; // rbx
  __int64 v26; // rcx
  char v27; // al
  _WORD *v28; // rdx
  __int64 v29; // rdx
  char v30; // r13
  __int64 v31; // rdi
  _BYTE *v32; // rax
  char *v33; // rax
  _BYTE *v34; // rax
  _BYTE *v35; // rax
  char *v36; // rax
  _BYTE *v37; // rax
  int v38; // edx
  __int64 v39; // r12
  __int64 v40; // rdi
  char *v41; // rax
  __int64 v42; // rdx
  _QWORD *v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r15
  _QWORD *v47; // rdx
  char *v48; // r14
  size_t v49; // rax
  _BYTE *v50; // rdi
  size_t v51; // r13
  _BYTE *v52; // rax
  _BYTE *v53; // r14
  __int64 v54; // rax
  _QWORD *v55; // rdx
  const char **v56; // r13
  char v57; // al
  __int64 v58; // rax
  _BYTE *v59; // rax
  _BYTE *v60; // rdx
  _BYTE *i; // rdx
  int v62; // r13d
  __int64 v63; // rax
  __int64 v64; // rbx
  __int64 v65; // r14
  char *v66; // rcx
  char v67; // al
  bool v68; // cf
  __int64 v69; // rax
  char v70; // al
  __int64 v71; // rax
  __int64 v73; // [rsp+28h] [rbp-268h]
  __int64 v74; // [rsp+30h] [rbp-260h]
  __int64 *v75; // [rsp+38h] [rbp-258h]
  __int64 v76; // [rsp+38h] [rbp-258h]
  char v77; // [rsp+38h] [rbp-258h]
  void *v78; // [rsp+40h] [rbp-250h] BYREF
  const char *v79; // [rsp+48h] [rbp-248h]
  char v80; // [rsp+50h] [rbp-240h]
  _QWORD v81[4]; // [rsp+60h] [rbp-230h] BYREF
  int v82; // [rsp+80h] [rbp-210h]
  _BYTE **v83; // [rsp+88h] [rbp-208h]
  _BYTE *v84; // [rsp+90h] [rbp-200h] BYREF
  __int64 v85; // [rsp+98h] [rbp-1F8h]
  _BYTE v86[64]; // [rsp+A0h] [rbp-1F0h] BYREF
  _BYTE *v87; // [rsp+E0h] [rbp-1B0h] BYREF
  __int64 v88; // [rsp+E8h] [rbp-1A8h]
  _BYTE v89[96]; // [rsp+F0h] [rbp-1A0h] BYREF
  _BYTE *v90; // [rsp+150h] [rbp-140h] BYREF
  __int64 v91; // [rsp+158h] [rbp-138h]
  _BYTE v92[304]; // [rsp+160h] [rbp-130h] BYREF

  v4 = a1;
  v6 = (__int64 *)(a1 + 592);
  if ( (*(_BYTE *)(a1 + 680) & 1) == 0 )
    v6 = sub_16E8D30();
  v90 = v92;
  v91 = 0x10000000000LL;
  v87 = v89;
  v83 = &v90;
  v88 = 0x400000000LL;
  v81[0] = &unk_49EFC48;
  v82 = 1;
  memset(&v81[1], 0, 24);
  sub_16E7A40((__int64)v81, 0, 0, 0);
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 296) + 16LL);
  if ( v7 )
  {
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, _BYTE **, __int64))(*(_QWORD *)v7 + 24LL))(
      v7,
      a2,
      v81,
      &v87,
      a3);
    v85 = 0x4000000000LL;
    v10 = v91;
    v84 = v86;
    v11 = 8LL * (unsigned int)v91;
    if ( v11 )
    {
      v59 = v86;
      v60 = v86;
      if ( v11 > 0x40 )
      {
        sub_16CD150((__int64)&v84, v86, 8LL * (unsigned int)v91, 1, v8, v9);
        v60 = v84;
        v59 = &v84[(unsigned int)v85];
      }
      for ( i = &v60[v11]; i != v59; ++v59 )
      {
        if ( v59 )
          *v59 = 0;
      }
      LODWORD(v85) = v11;
      v10 = v91;
    }
    v12 = (unsigned int)(8 * v10);
    v13 = 0;
    if ( (_DWORD)v12 )
    {
      do
        v84[v13++] = 0;
      while ( v12 != v13 );
    }
    v14 = v88;
    if ( (_DWORD)v88 )
    {
      v75 = v6;
      v15 = 1;
      v16 = v4;
      v17 = 0;
      while ( 1 )
      {
        v18 = &v87[v17];
        v19 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(v16 + 296) + 8LL) + 48LL))(
                *(_QWORD *)(*(_QWORD *)(v16 + 296) + 8LL),
                *(unsigned int *)&v87[v17 + 12]);
        v20 = 0;
        v8 = *(_DWORD *)(v19 + 12);
        if ( v8 )
        {
          do
          {
            v21 = (unsigned int)(v20 + *(_DWORD *)(v19 + 8) + 8 * *((_DWORD *)v18 + 2));
            ++v20;
            v84[v21] = v15;
          }
          while ( *(_DWORD *)(v19 + 12) != v20 );
        }
        v17 += 24;
        if ( v14 == v15 )
          break;
        ++v15;
      }
      v4 = v16;
      v6 = v75;
    }
    v22 = (void *)v6[3];
    if ( (unsigned __int64)(v6[2] - (_QWORD)v22) <= 0xA )
    {
      sub_16E7EE0((__int64)v6, "encoding: [", 0xBu);
    }
    else
    {
      qmemcpy(v22, "encoding: [", 11);
      v6[3] += 11;
    }
    if ( (_DWORD)v91 )
    {
      v23 = 0;
      v76 = (unsigned int)(v91 - 1);
      v24 = (__int64)v6;
      v25 = 0;
      while ( 2 )
      {
        LODWORD(v26) = v23 + 1;
        v27 = v84[v23];
        do
        {
          if ( v84[(unsigned int)v26] != v27 )
          {
LABEL_29:
            v28 = *(_WORD **)(v24 + 24);
            if ( *(_QWORD *)(v24 + 16) - (_QWORD)v28 <= 1u )
            {
              sub_16E7EE0(v24, "0b", 2u);
            }
            else
            {
              *v28 = 25136;
              *(_QWORD *)(v24 + 24) += 2LL;
            }
            v62 = 7;
            v63 = v25;
            v64 = v24;
            v65 = v63;
            do
            {
              while ( 1 )
              {
                v69 = (unsigned int)(v23 + 7 - v62);
                if ( *(_BYTE *)(*(_QWORD *)(v4 + 280) + 16LL) )
                  v69 = (unsigned int)(v23 + v62);
                v70 = v84[v69];
                if ( v70 )
                  break;
                sub_16E7A90(v64, ((char)v90[v65] >> v62) & 1);
                v68 = v62-- == 0;
                if ( v68 )
                  goto LABEL_100;
              }
              v66 = *(char **)(v64 + 24);
              v67 = v70 + 64;
              if ( (unsigned __int64)v66 >= *(_QWORD *)(v64 + 16) )
              {
                sub_16E7DE0(v64, v67);
              }
              else
              {
                *(_QWORD *)(v64 + 24) = v66 + 1;
                *v66 = v67;
              }
              v68 = v62-- == 0;
            }
            while ( !v68 );
LABEL_100:
            v71 = v65;
            v24 = v64;
            v25 = v71;
            goto LABEL_39;
          }
          v26 = (unsigned int)(v26 + 1);
        }
        while ( v23 + 8 != (_DWORD)v26 );
        if ( v27 == -1 )
          goto LABEL_29;
        if ( !v27 )
        {
          v57 = v90[v25];
          v79 = "0x%02x";
          v80 = v57;
          v78 = &unk_49EF3B0;
          sub_16E8450(v24, (__int64)&v78, (__int64)&unk_49EF3B0, v26, v8, v9);
          goto LABEL_39;
        }
        v29 = (unsigned __int8)v90[v25];
        v30 = v27 + 64;
        if ( (_BYTE)v29 )
        {
          v80 = v90[v25];
          v79 = "0x%02x";
          v78 = &unk_49EF3B0;
          v31 = sub_16E8450(v24, (__int64)&v78, v29, v26, v8, v9);
          v32 = *(_BYTE **)(v31 + 24);
          if ( (unsigned __int64)v32 >= *(_QWORD *)(v31 + 16) )
          {
            v31 = sub_16E7DE0(v31, 39);
            v33 = *(char **)(v31 + 24);
            if ( (unsigned __int64)v33 >= *(_QWORD *)(v31 + 16) )
              goto LABEL_82;
LABEL_37:
            *(_QWORD *)(v31 + 24) = v33 + 1;
            *v33 = v30;
            v34 = *(_BYTE **)(v31 + 24);
            if ( (unsigned __int64)v34 < *(_QWORD *)(v31 + 16) )
              goto LABEL_38;
LABEL_83:
            sub_16E7DE0(v31, 39);
          }
          else
          {
            *(_QWORD *)(v31 + 24) = v32 + 1;
            *v32 = 39;
            v33 = *(char **)(v31 + 24);
            if ( (unsigned __int64)v33 < *(_QWORD *)(v31 + 16) )
              goto LABEL_37;
LABEL_82:
            v31 = sub_16E7DE0(v31, v30);
            v34 = *(_BYTE **)(v31 + 24);
            if ( (unsigned __int64)v34 >= *(_QWORD *)(v31 + 16) )
              goto LABEL_83;
LABEL_38:
            *(_QWORD *)(v31 + 24) = v34 + 1;
            *v34 = 39;
          }
LABEL_39:
          if ( v76 == v25 )
            goto LABEL_46;
LABEL_40:
          if ( (_DWORD)v25 != -1 )
          {
            v35 = *(_BYTE **)(v24 + 24);
            if ( (unsigned __int64)v35 >= *(_QWORD *)(v24 + 16) )
            {
              sub_16E7DE0(v24, 44);
            }
            else
            {
              *(_QWORD *)(v24 + 24) = v35 + 1;
              *v35 = 44;
            }
          }
          ++v25;
          v23 += 8;
          continue;
        }
        break;
      }
      v36 = *(char **)(v24 + 24);
      if ( (unsigned __int64)v36 >= *(_QWORD *)(v24 + 16) )
      {
        sub_16E7DE0(v24, v30);
        goto LABEL_39;
      }
      *(_QWORD *)(v24 + 24) = v36 + 1;
      *v36 = v30;
      if ( v76 != v25 )
        goto LABEL_40;
LABEL_46:
      v6 = (__int64 *)v24;
    }
    v37 = (_BYTE *)v6[3];
    if ( (_BYTE *)v6[2] == v37 )
    {
      sub_16E7EE0((__int64)v6, "]", 1u);
      if ( (_DWORD)v88 )
        goto LABEL_49;
    }
    else
    {
      *v37 = 93;
      v38 = v88;
      ++v6[3];
      if ( v38 )
        goto LABEL_49;
    }
    if ( a4 )
    {
LABEL_15:
      if ( v84 != v86 )
        _libc_free((unsigned __int64)v84);
      goto LABEL_17;
    }
LABEL_49:
    sub_1263B40((__int64)v6, "\n");
    if ( (_DWORD)v88 )
    {
      v77 = 65;
      v39 = 0;
      v73 = v4;
      v74 = 24LL * (unsigned int)v88;
      while ( 1 )
      {
        v53 = &v87[v39];
        v54 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(v73 + 296) + 8LL) + 48LL))(
                *(_QWORD *)(*(_QWORD *)(v73 + 296) + 8LL),
                *(unsigned int *)&v87[v39 + 12]);
        v55 = (_QWORD *)v6[3];
        v56 = (const char **)v54;
        if ( (unsigned __int64)(v6[2] - (_QWORD)v55) > 7 )
        {
          v40 = (__int64)v6;
          *v55 = 0x2070757869662020LL;
          v41 = (char *)(v6[3] + 8);
          v6[3] = (__int64)v41;
          if ( (unsigned __int64)v41 < v6[2] )
            goto LABEL_52;
        }
        else
        {
          v40 = sub_16E7EE0((__int64)v6, "  fixup ", 8u);
          v41 = *(char **)(v40 + 24);
          if ( (unsigned __int64)v41 < *(_QWORD *)(v40 + 16) )
          {
LABEL_52:
            *(_QWORD *)(v40 + 24) = v41 + 1;
            *v41 = v77;
            goto LABEL_53;
          }
        }
        v40 = sub_16E7DE0(v40, v77);
LABEL_53:
        v42 = *(_QWORD *)(v40 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(v40 + 16) - v42) <= 2 )
        {
          v58 = sub_16E7EE0(v40, " - ", 3u);
          v43 = *(_QWORD **)(v58 + 24);
          v40 = v58;
        }
        else
        {
          *(_BYTE *)(v42 + 2) = 32;
          *(_WORD *)v42 = 11552;
          v43 = (_QWORD *)(*(_QWORD *)(v40 + 24) + 3LL);
          *(_QWORD *)(v40 + 24) = v43;
        }
        if ( *(_QWORD *)(v40 + 16) - (_QWORD)v43 <= 7u )
        {
          v40 = sub_16E7EE0(v40, "offset: ", 8u);
        }
        else
        {
          *v43 = 0x203A74657366666FLL;
          *(_QWORD *)(v40 + 24) += 8LL;
        }
        v44 = sub_16E7A90(v40, *((unsigned int *)v53 + 2));
        v45 = *(_QWORD *)(v44 + 24);
        v46 = v44;
        if ( (unsigned __int64)(*(_QWORD *)(v44 + 16) - v45) <= 8 )
        {
          v46 = sub_16E7EE0(v44, ", value: ", 9u);
        }
        else
        {
          *(_BYTE *)(v45 + 8) = 32;
          *(_QWORD *)v45 = 0x3A65756C6176202CLL;
          *(_QWORD *)(v44 + 24) += 9LL;
        }
        sub_38CDBE0(*(_QWORD *)v53, v46, 0);
        v47 = *(_QWORD **)(v46 + 24);
        if ( *(_QWORD *)(v46 + 16) - (_QWORD)v47 <= 7u )
        {
          v46 = sub_16E7EE0(v46, ", kind: ", 8u);
        }
        else
        {
          *v47 = 0x203A646E696B202CLL;
          *(_QWORD *)(v46 + 24) += 8LL;
        }
        v48 = (char *)*v56;
        if ( !*v56 )
          goto LABEL_74;
        v49 = strlen(*v56);
        v50 = *(_BYTE **)(v46 + 24);
        v51 = v49;
        v52 = *(_BYTE **)(v46 + 16);
        if ( v51 > v52 - v50 )
        {
          v46 = sub_16E7EE0(v46, v48, v51);
LABEL_74:
          v50 = *(_BYTE **)(v46 + 24);
          if ( *(_BYTE **)(v46 + 16) == v50 )
            goto LABEL_75;
          goto LABEL_66;
        }
        if ( v51 )
        {
          memcpy(v50, v48, v51);
          v52 = *(_BYTE **)(v46 + 16);
          v50 = (_BYTE *)(v51 + *(_QWORD *)(v46 + 24));
          *(_QWORD *)(v46 + 24) = v50;
        }
        if ( v52 == v50 )
        {
LABEL_75:
          sub_16E7EE0(v46, "\n", 1u);
          goto LABEL_67;
        }
LABEL_66:
        *v50 = 10;
        ++*(_QWORD *)(v46 + 24);
LABEL_67:
        ++v77;
        v39 += 24;
        if ( v39 == v74 )
          goto LABEL_15;
      }
    }
    goto LABEL_15;
  }
LABEL_17:
  v81[0] = &unk_49EFD28;
  sub_16E7960((__int64)v81);
  if ( v87 != v89 )
    _libc_free((unsigned __int64)v87);
  if ( v90 != v92 )
    _libc_free((unsigned __int64)v90);
}
