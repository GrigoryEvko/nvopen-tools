// Function: sub_E547B0
// Address: 0xe547b0
//
__int64 __fastcall sub_E547B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 *v5; // rbx
  __int64 result; // rax
  __int64 v7; // rdi
  __int64 v8; // r8
  __int64 v9; // r9
  int v10; // eax
  _BYTE *v11; // rdx
  unsigned __int64 v12; // r13
  _BYTE *v13; // rax
  _BYTE *i; // rdx
  __int64 v15; // rcx
  __int64 v16; // rax
  int v17; // r14d
  int v18; // r15d
  __int64 v19; // rbx
  __int64 v20; // r12
  _BYTE *v21; // r13
  __int64 v22; // rdx
  int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // r9
  int v27; // r13d
  __int64 v28; // r15
  __int64 v29; // rbx
  int v30; // esi
  __int64 v31; // r8
  char v32; // al
  int v33; // r14d
  char *v34; // rsi
  char v35; // al
  bool v36; // cf
  __int64 v37; // rax
  char v38; // al
  _BYTE *v39; // rax
  char *v40; // rcx
  __int64 v41; // rcx
  char v42; // r14
  __int64 v43; // rdi
  _BYTE *v44; // rax
  char *v45; // rax
  _BYTE *v46; // rax
  char *v47; // rsi
  __int64 v48; // r13
  __int64 v49; // rdi
  char *v50; // rax
  __int64 v51; // rdx
  _QWORD *v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rdx
  _QWORD *v55; // rdx
  __int64 v56; // r15
  size_t v57; // rax
  _BYTE *v58; // rdi
  size_t v59; // r14
  _BYTE *v60; // r15
  __int64 v61; // rax
  _QWORD *v62; // rdx
  const char **v63; // r14
  char *v64; // rax
  char v65; // al
  __int64 v66; // rax
  __int64 v67; // [rsp+18h] [rbp-258h]
  unsigned __int8 *src; // [rsp+20h] [rbp-250h]
  __int64 *v69; // [rsp+28h] [rbp-248h]
  __int64 v70; // [rsp+28h] [rbp-248h]
  char v71; // [rsp+28h] [rbp-248h]
  void *v72; // [rsp+30h] [rbp-240h] BYREF
  const char *v73; // [rsp+38h] [rbp-238h]
  char v74; // [rsp+40h] [rbp-230h]
  _BYTE *v75; // [rsp+50h] [rbp-220h] BYREF
  unsigned __int64 v76; // [rsp+58h] [rbp-218h]
  __int64 v77; // [rsp+60h] [rbp-210h]
  _BYTE v78[72]; // [rsp+68h] [rbp-208h] BYREF
  _BYTE *v79; // [rsp+B0h] [rbp-1C0h] BYREF
  __int64 v80; // [rsp+B8h] [rbp-1B8h]
  _BYTE v81[96]; // [rsp+C0h] [rbp-1B0h] BYREF
  _BYTE *v82; // [rsp+120h] [rbp-150h] BYREF
  __int64 v83; // [rsp+128h] [rbp-148h]
  __int64 v84; // [rsp+130h] [rbp-140h]
  _BYTE v85[312]; // [rsp+138h] [rbp-138h] BYREF

  v4 = a1;
  v5 = (__int64 *)(a1 + 640);
  if ( !*(_BYTE *)(a1 + 745) )
    v5 = sub_CB7330();
  v84 = 256;
  v82 = v85;
  v79 = v81;
  v80 = 0x400000000LL;
  result = *(_QWORD *)(a1 + 328);
  v83 = 0;
  v7 = *(_QWORD *)(result + 16);
  if ( !v7 )
    return result;
  (*(void (__fastcall **)(__int64, __int64, _BYTE **, _BYTE **, __int64))(*(_QWORD *)v7 + 24LL))(v7, a2, &v82, &v79, a3);
  v10 = v83;
  v11 = v78;
  v76 = 0;
  v75 = v78;
  v77 = 64;
  v12 = 8 * v83;
  if ( 8 * v83 )
  {
    v13 = v78;
    if ( v12 > 0x40 )
    {
      sub_C8D290((__int64)&v75, v78, 8 * v83, 1u, v8, v9);
      v11 = v75;
      v13 = &v75[v76];
    }
    for ( i = &v11[v12]; i != v13; ++v13 )
    {
      if ( v13 )
        *v13 = 0;
    }
    v76 = v12;
    v10 = v83;
  }
  v15 = (unsigned int)(8 * v10);
  v16 = 0;
  if ( (_DWORD)v15 )
  {
    do
      v75[v16++] = 0;
    while ( v15 != v16 );
  }
  v17 = v80;
  if ( (_DWORD)v80 )
  {
    v69 = v5;
    v18 = 1;
    v19 = v4;
    v20 = 0;
    while ( 1 )
    {
      v21 = &v79[v20];
      v22 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(v19 + 328) + 8LL) + 64LL))(
              *(_QWORD *)(*(_QWORD *)(v19 + 328) + 8LL),
              *(unsigned int *)&v79[v20 + 12]);
      v23 = 0;
      if ( *(_DWORD *)(v22 + 12) )
      {
        do
        {
          v24 = (unsigned int)(v23 + *(_DWORD *)(v22 + 8) + 8 * *((_DWORD *)v21 + 2));
          ++v23;
          v75[v24] = v18;
        }
        while ( *(_DWORD *)(v22 + 12) != v23 );
      }
      v20 += 24;
      if ( v17 == v18 )
        break;
      ++v18;
    }
    v4 = v19;
    v5 = v69;
  }
  sub_904010((__int64)v5, "encoding: [");
  if ( !(_DWORD)v83 )
    goto LABEL_56;
  v27 = 0;
  v70 = (unsigned int)(v83 - 1);
  v28 = (__int64)v5;
  v29 = 0;
  while ( 2 )
  {
    v30 = v27 + 1;
    v31 = (unsigned int)(v27 + 8);
    v32 = v75[v27];
    do
    {
      if ( v75[v30] != v32 )
      {
LABEL_32:
        v33 = 7;
        sub_904010(v28, "0b");
        do
        {
          while ( 1 )
          {
            v37 = (unsigned int)(v27 + 7 - v33);
            if ( *(_BYTE *)(*(_QWORD *)(v4 + 312) + 16LL) )
              v37 = (unsigned int)(v27 + v33);
            v38 = v75[v37];
            if ( v38 )
              break;
            sub_CB59D0(v28, ((char)v82[v29] >> v33) & 1);
            v36 = v33-- == 0;
            if ( v36 )
              goto LABEL_40;
          }
          v34 = *(char **)(v28 + 32);
          v35 = v38 + 64;
          if ( (unsigned __int64)v34 >= *(_QWORD *)(v28 + 24) )
          {
            sub_CB5D20(v28, v35);
          }
          else
          {
            *(_QWORD *)(v28 + 32) = v34 + 1;
            *v34 = v35;
          }
          v36 = v33-- == 0;
        }
        while ( !v36 );
        goto LABEL_40;
      }
      ++v30;
    }
    while ( (_DWORD)v31 != v30 );
    if ( v32 == -1 )
      goto LABEL_32;
    v40 = &v82[v29];
    if ( !v32 )
    {
      v65 = *v40;
      v73 = "0x%02x";
      v74 = v65;
      v72 = &unk_49DD0D8;
      sub_CB6620(v28, (__int64)&v72, (__int64)"0x%02x", (__int64)&unk_49DD0D8, v31, v26);
LABEL_40:
      if ( v70 == v29 )
        goto LABEL_55;
LABEL_41:
      if ( (_DWORD)v29 != -1 )
      {
        v39 = *(_BYTE **)(v28 + 32);
        if ( (unsigned __int64)v39 >= *(_QWORD *)(v28 + 24) )
        {
          sub_CB5D20(v28, 44);
        }
        else
        {
          *(_QWORD *)(v28 + 32) = v39 + 1;
          *v39 = 44;
        }
      }
      ++v29;
      v27 += 8;
      continue;
    }
    break;
  }
  v41 = (unsigned __int8)*v40;
  v42 = v32 + 64;
  if ( !(_BYTE)v41 )
  {
    v64 = *(char **)(v28 + 32);
    if ( (unsigned __int64)v64 >= *(_QWORD *)(v28 + 24) )
    {
      sub_CB5D20(v28, v42);
    }
    else
    {
      *(_QWORD *)(v28 + 32) = v64 + 1;
      *v64 = v42;
    }
    goto LABEL_40;
  }
  v74 = v41;
  v73 = "0x%02x";
  v72 = &unk_49DD0D8;
  v43 = sub_CB6620(v28, (__int64)&v72, v25, v41, v31, v26);
  v44 = *(_BYTE **)(v43 + 32);
  if ( (unsigned __int64)v44 >= *(_QWORD *)(v43 + 24) )
  {
    v43 = sub_CB5D20(v43, 39);
  }
  else
  {
    *(_QWORD *)(v43 + 32) = v44 + 1;
    *v44 = 39;
  }
  v45 = *(char **)(v43 + 32);
  if ( (unsigned __int64)v45 >= *(_QWORD *)(v43 + 24) )
  {
    v43 = sub_CB5D20(v43, v42);
  }
  else
  {
    *(_QWORD *)(v43 + 32) = v45 + 1;
    *v45 = v42;
  }
  v46 = *(_BYTE **)(v43 + 32);
  if ( (unsigned __int64)v46 >= *(_QWORD *)(v43 + 24) )
  {
    sub_CB5D20(v43, 39);
    goto LABEL_40;
  }
  *(_QWORD *)(v43 + 32) = v46 + 1;
  *v46 = 39;
  if ( v70 != v29 )
    goto LABEL_41;
LABEL_55:
  v5 = (__int64 *)v28;
LABEL_56:
  v47 = "]\n";
  sub_904010((__int64)v5, "]\n");
  result = (unsigned int)v80;
  if ( (_DWORD)v80 )
  {
    v71 = 65;
    v48 = 0;
    v67 = 24LL * (unsigned int)v80;
    do
    {
      v60 = &v79[v48];
      v61 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(v4 + 328) + 8LL) + 64LL))(
              *(_QWORD *)(*(_QWORD *)(v4 + 328) + 8LL),
              *(unsigned int *)&v79[v48 + 12]);
      v62 = (_QWORD *)v5[4];
      v63 = (const char **)v61;
      if ( (unsigned __int64)(v5[3] - (_QWORD)v62) > 7 )
      {
        v49 = (__int64)v5;
        *v62 = 0x2070757869662020LL;
        v50 = (char *)(v5[4] + 8);
        v5[4] = (__int64)v50;
        if ( v5[3] > (unsigned __int64)v50 )
          goto LABEL_59;
      }
      else
      {
        v49 = sub_CB6200((__int64)v5, "  fixup ", 8u);
        v50 = *(char **)(v49 + 32);
        if ( *(_QWORD *)(v49 + 24) > (unsigned __int64)v50 )
        {
LABEL_59:
          *(_QWORD *)(v49 + 32) = v50 + 1;
          *v50 = v71;
          goto LABEL_60;
        }
      }
      v49 = sub_CB5D20(v49, v71);
LABEL_60:
      v51 = *(_QWORD *)(v49 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v49 + 24) - v51) <= 2 )
      {
        v66 = sub_CB6200(v49, (unsigned __int8 *)" - ", 3u);
        v52 = *(_QWORD **)(v66 + 32);
        v49 = v66;
      }
      else
      {
        *(_BYTE *)(v51 + 2) = 32;
        *(_WORD *)v51 = 11552;
        v52 = (_QWORD *)(*(_QWORD *)(v49 + 32) + 3LL);
        *(_QWORD *)(v49 + 32) = v52;
      }
      if ( *(_QWORD *)(v49 + 24) - (_QWORD)v52 <= 7u )
      {
        v49 = sub_CB6200(v49, (unsigned __int8 *)"offset: ", 8u);
      }
      else
      {
        *v52 = 0x203A74657366666FLL;
        *(_QWORD *)(v49 + 32) += 8LL;
      }
      v53 = sub_CB59D0(v49, *((unsigned int *)v60 + 2));
      v54 = *(_QWORD *)(v53 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v53 + 24) - v54) <= 8 )
      {
        sub_CB6200(v53, ", value: ", 9u);
      }
      else
      {
        *(_BYTE *)(v54 + 8) = 32;
        *(_QWORD *)v54 = 0x3A65756C6176202CLL;
        *(_QWORD *)(v53 + 32) += 9LL;
      }
      sub_E7FAD0(*(_QWORD *)v60, v5, *(_QWORD *)(v4 + 312), 0);
      v55 = (_QWORD *)v5[4];
      if ( (unsigned __int64)(v5[3] - (_QWORD)v55) <= 7 )
      {
        v56 = sub_CB6200((__int64)v5, ", kind: ", 8u);
      }
      else
      {
        v56 = (__int64)v5;
        *v55 = 0x203A646E696B202CLL;
        v5[4] += 8;
      }
      v47 = (char *)*v63;
      if ( !*v63 )
        goto LABEL_83;
      src = (unsigned __int8 *)*v63;
      v57 = strlen(*v63);
      v58 = *(_BYTE **)(v56 + 32);
      v47 = (char *)src;
      v59 = v57;
      result = *(_QWORD *)(v56 + 24);
      if ( v59 > result - (__int64)v58 )
      {
        v56 = sub_CB6200(v56, src, v59);
LABEL_83:
        result = *(_QWORD *)(v56 + 24);
        v58 = *(_BYTE **)(v56 + 32);
        if ( (_BYTE *)result == v58 )
          goto LABEL_84;
        goto LABEL_73;
      }
      if ( v59 )
      {
        memcpy(v58, src, v59);
        result = *(_QWORD *)(v56 + 24);
        v58 = (_BYTE *)(v59 + *(_QWORD *)(v56 + 32));
        *(_QWORD *)(v56 + 32) = v58;
      }
      if ( (_BYTE *)result == v58 )
      {
LABEL_84:
        v47 = "\n";
        result = sub_CB6200(v56, (unsigned __int8 *)"\n", 1u);
        goto LABEL_74;
      }
LABEL_73:
      *v58 = 10;
      ++*(_QWORD *)(v56 + 32);
LABEL_74:
      ++v71;
      v48 += 24;
    }
    while ( v67 != v48 );
  }
  if ( v75 != v78 )
    result = _libc_free(v75, v47);
  if ( v79 != v81 )
    result = _libc_free(v79, v47);
  if ( v82 != v85 )
    return _libc_free(v82, v47);
  return result;
}
