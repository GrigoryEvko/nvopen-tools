// Function: sub_310C780
// Address: 0x310c780
//
__int64 __fastcall sub_310C780(__int64 a1, _DWORD *a2, __int64 a3)
{
  __int64 v4; // r13
  void *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdi
  _BYTE *v11; // rax
  __int64 v12; // rbx
  const char *v13; // rax
  size_t v14; // rdx
  _DWORD *v15; // rdi
  unsigned __int8 *v16; // rsi
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdi
  _BYTE *v22; // rax
  __int64 v23; // r12
  __int64 v24; // r13
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v28; // rax
  _DWORD *v29; // rdx
  __int64 v30; // r13
  __int64 v31; // rdx
  const char *v32; // rax
  size_t v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r9
  _DWORD *v36; // rdi
  unsigned __int8 *v37; // rsi
  unsigned __int64 v38; // rax
  __int64 v39; // r8
  __int64 v40; // rcx
  __int64 v41; // rdi
  __int64 v42; // r8
  __int64 v43; // r9
  _BYTE *v44; // rax
  unsigned __int64 v45; // r13
  unsigned __int64 v46; // r12
  unsigned __int64 v47; // rdi
  _DWORD *v48; // rax
  _DWORD *v49; // r14
  _DWORD *v50; // r12
  unsigned int v51; // r13d
  unsigned int v52; // ebx
  __int64 v53; // rdi
  __int64 v54; // rdx
  __m128i si128; // xmm0
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rdi
  __int64 v59; // rdi
  _BYTE *v60; // rax
  __int64 v61; // rax
  _WORD *v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // [rsp+8h] [rbp-98h]
  size_t v67; // [rsp+18h] [rbp-88h]
  size_t v68; // [rsp+18h] [rbp-88h]
  __int64 v69; // [rsp+20h] [rbp-80h]
  __int64 (__fastcall *v71)(void **, __int64, int); // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v72; // [rsp+38h] [rbp-68h]
  unsigned __int64 v73; // [rsp+40h] [rbp-60h]
  void *v74; // [rsp+50h] [rbp-50h] BYREF
  char *v75; // [rsp+58h] [rbp-48h]
  __int64 (__fastcall *v76)(void **, __int64, int); // [rsp+60h] [rbp-40h]
  __int64 (__fastcall *v77)(__int64, _QWORD *, unsigned int *); // [rsp+68h] [rbp-38h]

  v4 = *(_QWORD *)a2;
  v5 = *(void **)(*(_QWORD *)a2 + 32LL);
  if ( *(_QWORD *)(*(_QWORD *)a2 + 24LL) - (_QWORD)v5 <= 0xCu )
  {
    v4 = sub_CB6200(v4, "Module Hash: ", 0xDu);
  }
  else
  {
    qmemcpy(v5, "Module Hash: ", 13);
    *(_QWORD *)(v4 + 32) += 13LL;
  }
  v6 = sub_3147DF0(a3, a2[2] != 0);
  v75 = "%016lx";
  v76 = (__int64 (__fastcall *)(void **, __int64, int))v6;
  v74 = &unk_49DC5C0;
  v10 = sub_CB6620(v4, (__int64)&v74, (__int64)&unk_49DC5C0, v7, v8, v9);
  v11 = *(_BYTE **)(v10 + 32);
  if ( *(_BYTE **)(v10 + 24) == v11 )
  {
    sub_CB6200(v10, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v11 = 10;
    ++*(_QWORD *)(v10 + 32);
  }
  v12 = *(_QWORD *)(a3 + 32);
  v69 = a3 + 24;
  if ( v12 == a3 + 24 )
    goto LABEL_25;
  while ( 1 )
  {
LABEL_16:
    v23 = v12 - 56;
    if ( !v12 )
      v23 = 0;
    if ( sub_B2FC80(v23) )
      goto LABEL_15;
    if ( a2[2] == 2 )
      break;
    v24 = *(_QWORD *)a2;
    v25 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
    if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a2 + 24LL) - v25) > 8 )
    {
      *(_BYTE *)(v25 + 8) = 32;
      *(_QWORD *)v25 = 0x6E6F6974636E7546LL;
      *(_QWORD *)(v24 + 32) += 9LL;
    }
    else
    {
      v24 = sub_CB6200(*(_QWORD *)a2, (unsigned __int8 *)"Function ", 9u);
    }
    v13 = sub_BD5D20(v23);
    v15 = *(_DWORD **)(v24 + 32);
    v16 = (unsigned __int8 *)v13;
    v17 = *(_QWORD *)(v24 + 24) - (_QWORD)v15;
    if ( v14 > v17 )
    {
      v26 = sub_CB6200(v24, v16, v14);
      v15 = *(_DWORD **)(v26 + 32);
      v24 = v26;
      if ( *(_QWORD *)(v26 + 24) - (_QWORD)v15 <= 6u )
        goto LABEL_23;
    }
    else
    {
      if ( v14 )
      {
        v67 = v14;
        memcpy(v15, v16, v14);
        v28 = *(_QWORD *)(v24 + 24);
        v29 = (_DWORD *)(*(_QWORD *)(v24 + 32) + v67);
        *(_QWORD *)(v24 + 32) = v29;
        v15 = v29;
        v17 = v28 - (_QWORD)v29;
      }
      if ( v17 <= 6 )
      {
LABEL_23:
        v24 = sub_CB6200(v24, (unsigned __int8 *)" Hash: ", 7u);
        goto LABEL_13;
      }
    }
    *v15 = 1935755296;
    *((_WORD *)v15 + 2) = 14952;
    *((_BYTE *)v15 + 6) = 32;
    *(_QWORD *)(v24 + 32) += 7LL;
LABEL_13:
    v18 = sub_3148040(v23, a2[2] == 1);
    v75 = "%016lx";
    v76 = (__int64 (__fastcall *)(void **, __int64, int))v18;
    v74 = &unk_49DC5C0;
    v21 = sub_CB6620(v24, (__int64)&v74, (__int64)&unk_49DC5C0, (__int64)&unk_49DC5B0, v19, v20);
    v22 = *(_BYTE **)(v21 + 32);
    if ( *(_BYTE **)(v21 + 24) != v22 )
    {
      *v22 = 10;
      ++*(_QWORD *)(v21 + 32);
      goto LABEL_15;
    }
    sub_CB6200(v21, (unsigned __int8 *)"\n", 1u);
    v12 = *(_QWORD *)(v12 + 8);
    if ( v69 == v12 )
      goto LABEL_25;
  }
  v77 = sub_310C730;
  v76 = (__int64 (__fastcall *)(void **, __int64, int))sub_310C720;
  sub_3147BA0(&v71, v23, &v74);
  if ( v76 )
    v76(&v74, (__int64)&v74, 3);
  v30 = *(_QWORD *)a2;
  v31 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a2 + 24LL) - v31) <= 8 )
  {
    v30 = sub_CB6200(*(_QWORD *)a2, (unsigned __int8 *)"Function ", 9u);
  }
  else
  {
    *(_BYTE *)(v31 + 8) = 32;
    *(_QWORD *)v31 = 0x6E6F6974636E7546LL;
    *(_QWORD *)(v30 + 32) += 9LL;
  }
  v32 = sub_BD5D20(v23);
  v36 = *(_DWORD **)(v30 + 32);
  v37 = (unsigned __int8 *)v32;
  v38 = *(_QWORD *)(v30 + 24) - (_QWORD)v36;
  if ( v33 > v38 )
  {
    v63 = sub_CB6200(v30, v37, v33);
    v36 = *(_DWORD **)(v63 + 32);
    v30 = v63;
    v38 = *(_QWORD *)(v63 + 24) - (_QWORD)v36;
  }
  else if ( v33 )
  {
    v68 = v33;
    memcpy(v36, v37, v33);
    v64 = *(_QWORD *)(v30 + 24);
    v33 = *(_QWORD *)(v30 + 32) + v68;
    *(_QWORD *)(v30 + 32) = v33;
    v36 = (_DWORD *)v33;
    v38 = v64 - v33;
  }
  if ( v38 <= 6 )
  {
    v30 = sub_CB6200(v30, (unsigned __int8 *)" Hash: ", 7u);
  }
  else
  {
    v39 = 14952;
    *v36 = 1935755296;
    *((_WORD *)v36 + 2) = 14952;
    *((_BYTE *)v36 + 6) = 32;
    *(_QWORD *)(v30 + 32) += 7LL;
  }
  v75 = "%016lx";
  v74 = &unk_49DC5C0;
  v76 = v71;
  v41 = sub_CB6620(v30, (__int64)&v74, v33, v34, v39, v35);
  v44 = *(_BYTE **)(v41 + 32);
  if ( *(_BYTE **)(v41 + 24) == v44 )
  {
    sub_CB6200(v41, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v44 = 10;
    ++*(_QWORD *)(v41 + 32);
  }
  v45 = v73;
  if ( !*(_DWORD *)(v73 + 16) )
    goto LABEL_38;
  v48 = *(_DWORD **)(v73 + 8);
  v49 = &v48[4 * *(unsigned int *)(v73 + 24)];
  if ( v48 == v49 )
    goto LABEL_38;
  while ( 2 )
  {
    v50 = v48;
    if ( *v48 == -1 )
    {
      if ( v48[1] != -1 )
        break;
      goto LABEL_78;
    }
    if ( *v48 == -2 && v48[1] == -2 )
    {
LABEL_78:
      v48 += 4;
      if ( v49 == v48 )
        goto LABEL_38;
      continue;
    }
    break;
  }
  if ( v49 == v48 )
    goto LABEL_38;
  v65 = v12;
  v51 = *v48;
LABEL_54:
  v52 = v50[1];
  v53 = *(_QWORD *)a2;
  v54 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a2 + 24LL) - v54) <= 0x16 )
  {
    v53 = sub_CB6200(v53, "\tIgnored Operand Hash: ", 0x17u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_44CF0E0);
    *(_DWORD *)(v54 + 16) = 1935755296;
    *(_WORD *)(v54 + 20) = 14952;
    *(_BYTE *)(v54 + 22) = 32;
    *(__m128i *)v54 = si128;
    *(_QWORD *)(v53 + 32) += 23LL;
  }
  v75 = "%016lx";
  v74 = &unk_49DC5C0;
  v76 = (__int64 (__fastcall *)(void **, __int64, int))*((_QWORD *)v50 + 1);
  v56 = sub_CB6620(v53, (__int64)&v74, v54, v40, v42, v43);
  v57 = *(_QWORD *)(v56 + 32);
  v58 = v56;
  if ( (unsigned __int64)(*(_QWORD *)(v56 + 24) - v57) <= 4 )
  {
    v58 = sub_CB6200(v56, " at (", 5u);
  }
  else
  {
    *(_DWORD *)v57 = 544497952;
    *(_BYTE *)(v57 + 4) = 40;
    *(_QWORD *)(v56 + 32) += 5LL;
  }
  v59 = sub_CB59D0(v58, v51);
  v60 = *(_BYTE **)(v59 + 32);
  if ( *(_BYTE **)(v59 + 24) == v60 )
  {
    v59 = sub_CB6200(v59, (unsigned __int8 *)",", 1u);
  }
  else
  {
    *v60 = 44;
    ++*(_QWORD *)(v59 + 32);
  }
  v61 = sub_CB59D0(v59, v52);
  v62 = *(_WORD **)(v61 + 32);
  if ( *(_QWORD *)(v61 + 24) - (_QWORD)v62 <= 1u )
  {
    sub_CB6200(v61, (unsigned __int8 *)")\n", 2u);
  }
  else
  {
    v40 = 2601;
    *v62 = 2601;
    *(_QWORD *)(v61 + 32) += 2LL;
  }
  for ( v50 += 4; v49 != v50; v50 += 4 )
  {
    if ( *v50 == -1 )
    {
      if ( v50[1] != -1 )
        goto LABEL_52;
    }
    else if ( *v50 != -2 || v50[1] != -2 )
    {
LABEL_52:
      if ( v49 == v50 )
        break;
      v51 = *v50;
      goto LABEL_54;
    }
  }
  v45 = v73;
  v12 = v65;
  if ( v73 )
  {
LABEL_38:
    sub_C7D6A0(*(_QWORD *)(v45 + 8), 16LL * *(unsigned int *)(v45 + 24), 8);
    j_j___libc_free_0(v45);
  }
  v46 = v72;
  if ( v72 )
  {
    v47 = *(_QWORD *)(v72 + 32);
    if ( v47 != v72 + 48 )
      _libc_free(v47);
    sub_C7D6A0(*(_QWORD *)(v46 + 8), 8LL * *(unsigned int *)(v46 + 24), 4);
    j_j___libc_free_0(v46);
  }
LABEL_15:
  v12 = *(_QWORD *)(v12 + 8);
  if ( v69 != v12 )
    goto LABEL_16;
LABEL_25:
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
