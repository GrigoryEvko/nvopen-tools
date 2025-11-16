// Function: sub_EC74A0
// Address: 0xec74a0
//
__int64 __fastcall sub_EC74A0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdi
  char *v5; // r14
  __int64 v6; // rcx
  unsigned int v7; // r10d
  __int64 v8; // rdi
  unsigned int v9; // r12d
  const void *v11; // r8
  char *v12; // r13
  char *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rdi
  char *v19; // rax
  char *v20; // rdi
  unsigned __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // r15
  void (__fastcall *v24)(__int64, unsigned __int64, _QWORD); // r14
  __int64 v25; // rdi
  char v26; // al
  unsigned __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // r8
  const char *v31; // r9
  size_t v32; // rax
  _BYTE *v33; // rax
  size_t v34; // rdx
  char *v35; // rax
  char *v36; // r15
  char *v37; // rcx
  __int64 v38; // rdi
  void (__fastcall *v39)(__int64, char *, _QWORD *, char *, char *); // rax
  __int64 v40; // rdi
  void (__fastcall *v41)(__int64, char *, _QWORD *, char *, char *); // rax
  __int64 v42; // [rsp+0h] [rbp-170h]
  const char *v43; // [rsp+8h] [rbp-168h]
  __int64 v44; // [rsp+10h] [rbp-160h]
  __int64 v45; // [rsp+18h] [rbp-158h]
  const char *v46; // [rsp+18h] [rbp-158h]
  char *v47; // [rsp+18h] [rbp-158h]
  void *srcc; // [rsp+20h] [rbp-150h]
  _QWORD *srcd; // [rsp+20h] [rbp-150h]
  _BOOL4 src; // [rsp+20h] [rbp-150h]
  char *srca; // [rsp+20h] [rbp-150h]
  void *srcb; // [rsp+20h] [rbp-150h]
  char *srce; // [rsp+20h] [rbp-150h]
  char v54; // [rsp+3Fh] [rbp-131h] BYREF
  int v55; // [rsp+40h] [rbp-130h] BYREF
  int v56; // [rsp+44h] [rbp-12Ch] BYREF
  __int64 v57; // [rsp+48h] [rbp-128h] BYREF
  _BYTE *v58; // [rsp+50h] [rbp-120h] BYREF
  size_t n; // [rsp+58h] [rbp-118h]
  _WORD *v60; // [rsp+60h] [rbp-110h] BYREF
  __int64 v61; // [rsp+68h] [rbp-108h]
  char *v62; // [rsp+70h] [rbp-100h] BYREF
  void *v63; // [rsp+78h] [rbp-F8h]
  char *v64; // [rsp+80h] [rbp-F0h] BYREF
  char *v65; // [rsp+88h] [rbp-E8h]
  _QWORD v66[2]; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v67[2]; // [rsp+A0h] [rbp-D0h] BYREF
  const char *v68; // [rsp+B0h] [rbp-C0h]
  __int64 v69; // [rsp+B8h] [rbp-B8h]
  __int16 v70; // [rsp+C0h] [rbp-B0h]
  _QWORD v71[2]; // [rsp+D0h] [rbp-A0h] BYREF
  char *v72; // [rsp+E0h] [rbp-90h] BYREF
  __int16 v73; // [rsp+F0h] [rbp-80h]
  __int64 v74[2]; // [rsp+100h] [rbp-70h] BYREF
  _QWORD v75[2]; // [rsp+110h] [rbp-60h] BYREF
  __int64 v76; // [rsp+120h] [rbp-50h]
  __int64 v77; // [rsp+128h] [rbp-48h]
  __int64 v78; // [rsp+130h] [rbp-40h]

  v2 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v3 = sub_ECD690(v2);
  v4 = *(_QWORD *)(a1 + 8);
  v58 = 0;
  n = 0;
  v5 = (char *)v3;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, _BYTE **))(*(_QWORD *)v4 + 192LL))(v4, &v58) )
  {
    v18 = *(_QWORD *)(a1 + 8);
    v74[0] = (__int64)"expected identifier after '.section' directive";
    LOWORD(v76) = 259;
    return (unsigned int)sub_ECDA70(v18, v5, v74, 0, 0);
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
  {
    v8 = *(_QWORD *)(a1 + 8);
    v74[0] = (__int64)"unexpected token in '.section' directive";
    LOWORD(v76) = 259;
    return (unsigned int)sub_ECE0E0(v8, v74, 0, 0);
  }
  v11 = v58;
  v12 = (char *)n;
  v64 = (char *)v66;
  LOBYTE(v7) = v58 == 0 && &v58[n] != 0;
  v9 = v7;
  if ( (_BYTE)v7 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v74[0] = n;
  if ( n > 0xF )
  {
    srcc = v58;
    v19 = (char *)sub_22409D0(&v64, v74, 0);
    v11 = srcc;
    v64 = v19;
    v20 = v19;
    v66[0] = v74[0];
LABEL_16:
    memcpy(v20, v11, (size_t)v12);
    v12 = (char *)v74[0];
    v13 = v64;
    goto LABEL_9;
  }
  if ( n == 1 )
  {
    LOBYTE(v66[0]) = *v58;
    v13 = (char *)v66;
    goto LABEL_9;
  }
  if ( n )
  {
    v20 = (char *)v66;
    goto LABEL_16;
  }
  v13 = (char *)v66;
LABEL_9:
  v65 = v12;
  v12[(_QWORD)v13] = 0;
  if ( v65 == (char *)0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v64, ",", 1, v6);
  v14 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v15 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v14 + 24LL))(v14);
  sub_2241130(&v64, v65, 0, v15, v16);
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v60 = 0;
    v61 = 0;
    v62 = 0;
    v63 = 0;
    sub_E95100(&v57, v64, v65, (void **)&v60, (void **)&v62, &v56, &v54, &v55);
    v21 = v57 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v57 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v57 = 0;
      v67[0] = v21 | 1;
      sub_C64870((__int64)v71, v67);
      v28 = *(_QWORD *)(a1 + 8);
      v29 = (__int64)v5;
      v74[0] = (__int64)v71;
      LOWORD(v76) = 260;
      v9 = sub_ECDA70(v28, v5, v74, 0, 0);
      if ( (char **)v71[0] != &v72 )
      {
        v29 = (__int64)(v72 + 1);
        j_j___libc_free_0(v71[0], v72 + 1);
      }
      if ( (v67[0] & 1) != 0 || (v67[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(v67, v29);
      if ( (v57 & 1) != 0 || (v57 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v57, v29);
      goto LABEL_12;
    }
    v22 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v74[0] = (__int64)v75;
    srcd = (_QWORD *)v22;
    sub_EC5090(v74, *(_BYTE **)(v22 + 24), *(_QWORD *)(v22 + 24) + *(_QWORD *)(v22 + 32));
    v76 = srcd[7];
    v77 = srcd[8];
    v78 = srcd[9];
    if ( (((_DWORD)v76 - 22) & 0xFFFFFFFD) == 0 )
      goto LABEL_19;
    v30 = (__int64)v63;
    v31 = v62;
    if ( v63 == (void *)13 )
    {
      if ( *(_QWORD *)v62 == 0x6F63747865745F5FLL && *((_DWORD *)v62 + 2) == 1851747425 && v62[12] == 116 )
      {
        v42 = 6;
        v43 = "__text";
LABEL_40:
        if ( v5 && (v45 = (__int64)v63, srca = v62, v32 = strlen(v5), v31 = srca, v30 = v45, v32) )
        {
          v44 = v45;
          v46 = srca;
          srcb = (void *)v32;
          v33 = memchr(v5, 44, v32);
          v31 = v46;
          v30 = v44;
          if ( v33 && (v21 = v33 - v5 + 1, (unsigned __int64)srcb <= v21) )
          {
            v37 = v33 + 1;
            v36 = v5 - 1;
          }
          else
          {
            v34 = (size_t)srcb - v21;
            srce = &v5[v21];
            v35 = (char *)memchr(&v5[v21], 44, v34);
            v30 = v44;
            v31 = v46;
            v36 = v35;
            v37 = srce;
            if ( !v35 )
              v36 = v5 - 1;
          }
        }
        else
        {
          v36 = v5 - 1;
          v37 = v5;
        }
        v38 = *(_QWORD *)(a1 + 8);
        v47 = v37;
        v39 = *(void (__fastcall **)(__int64, char *, _QWORD *, char *, char *))(*(_QWORD *)v38 + 168LL);
        v70 = 1283;
        v67[0] = (__int64)"section \"";
        v68 = v31;
        v72 = "\" is deprecated";
        v69 = v30;
        v71[0] = v67;
        v73 = 770;
        v39(v38, v5, v71, v37, v36);
        v40 = *(_QWORD *)(a1 + 8);
        v41 = *(void (__fastcall **)(__int64, char *, _QWORD *, char *, char *))(*(_QWORD *)v40 + 160LL);
        v67[0] = (__int64)"change section name to \"";
        v71[0] = v67;
        v70 = 1283;
        v68 = v43;
        v73 = 770;
        v69 = v42;
        v72 = "\"";
        v41(v40, v5, v71, v47, v36);
        goto LABEL_19;
      }
      if ( *(_QWORD *)v62 == 0x6F63617461645F5FLL && *((_DWORD *)v62 + 2) == 1851747425 && v62[12] == 116 )
      {
        v42 = 6;
        v43 = "__data";
        goto LABEL_40;
      }
    }
    else if ( v63 == (void *)12 && *(_QWORD *)v62 == 0x5F74736E6F635F5FLL && *((_DWORD *)v62 + 2) == 1818324835 )
    {
      v42 = 7;
      v43 = "__const";
      goto LABEL_40;
    }
LABEL_19:
    if ( v61 == 6 )
    {
      src = *(_DWORD *)v60 != 1163157343 || v60[2] != 21592;
      v23 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      v24 = *(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)v23 + 176LL);
      v25 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v26 = !src ? 2 : 19;
    }
    else
    {
      v23 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      v24 = *(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)v23 + 176LL);
      v25 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v26 = 19;
    }
    v27 = sub_E6D970(v25, (__int64)v60, v61, v62, v63, v56, v55, v26, 0);
    v24(v23, v27, 0);
    if ( (_QWORD *)v74[0] != v75 )
      j_j___libc_free_0(v74[0], v75[0] + 1LL);
    goto LABEL_12;
  }
  v17 = *(_QWORD *)(a1 + 8);
  v74[0] = (__int64)"unexpected token in '.section' directive";
  LOWORD(v76) = 259;
  v9 = sub_ECE0E0(v17, v74, 0, 0);
LABEL_12:
  if ( v64 != (char *)v66 )
    j_j___libc_free_0(v64, v66[0] + 1LL);
  return v9;
}
