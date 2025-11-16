// Function: sub_39039A0
// Address: 0x39039a0
//
__int64 __fastcall sub_39039A0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdi
  char *v5; // r13
  unsigned int v6; // eax
  unsigned int v7; // r12d
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdi
  const void *v12; // r8
  char *v13; // r15
  char *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  _BYTE *v17; // rax
  size_t v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdi
  __int64 v22; // rax
  const char *v23; // rdx
  __int64 v24; // rax
  size_t v25; // rax
  _BYTE *v26; // rax
  size_t v27; // rdx
  unsigned __int64 v28; // rax
  char *v29; // rcx
  char *v30; // r8
  __int64 v31; // rdi
  void (__fastcall *v32)(__int64, char *); // rax
  __int64 v33; // rdi
  void (__fastcall *v34)(__int64, char *, char ***, char *, char *); // rax
  __int64 (*v35)(void); // rax
  __int64 v36; // r14
  void (__fastcall *v37)(__int64, __int64, _QWORD); // r13
  __int64 v38; // rdi
  int v39; // eax
  __int64 v40; // rax
  char *v41; // rax
  char *v42; // rdi
  __int64 v43; // rdi
  char *v44; // rax
  char *v45; // [rsp+0h] [rbp-160h]
  char *v46; // [rsp+8h] [rbp-158h]
  _QWORD *srca; // [rsp+10h] [rbp-150h]
  void *src; // [rsp+10h] [rbp-150h]
  void *srcb; // [rsp+10h] [rbp-150h]
  char *srcc; // [rsp+10h] [rbp-150h]
  char v51; // [rsp+27h] [rbp-139h] BYREF
  int v52; // [rsp+28h] [rbp-138h] BYREF
  int v53; // [rsp+2Ch] [rbp-134h] BYREF
  _BYTE *v54; // [rsp+30h] [rbp-130h] BYREF
  size_t n; // [rsp+38h] [rbp-128h]
  _WORD *v56; // [rsp+40h] [rbp-120h] BYREF
  unsigned __int64 v57; // [rsp+48h] [rbp-118h]
  _DWORD *v58; // [rsp+50h] [rbp-110h] BYREF
  unsigned __int64 v59; // [rsp+58h] [rbp-108h]
  _QWORD v60[2]; // [rsp+60h] [rbp-100h] BYREF
  char *v61; // [rsp+70h] [rbp-F0h] BYREF
  _QWORD *v62; // [rsp+78h] [rbp-E8h]
  __int16 v63; // [rsp+80h] [rbp-E0h]
  char **v64; // [rsp+90h] [rbp-D0h] BYREF
  char *v65; // [rsp+98h] [rbp-C8h]
  __int16 v66; // [rsp+A0h] [rbp-C0h]
  char *v67; // [rsp+B0h] [rbp-B0h] BYREF
  char *v68; // [rsp+B8h] [rbp-A8h]
  _QWORD v69[2]; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v70[2]; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v71; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v72[2]; // [rsp+F0h] [rbp-70h] BYREF
  _WORD v73[8]; // [rsp+100h] [rbp-60h] BYREF
  __int64 v74; // [rsp+110h] [rbp-50h]
  __int64 v75; // [rsp+118h] [rbp-48h]
  __int64 v76; // [rsp+120h] [rbp-40h]

  v2 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v3 = sub_3909290(v2);
  v4 = *(_QWORD *)(a1 + 8);
  v54 = 0;
  n = 0;
  v5 = (char *)v3;
  v6 = (*(__int64 (__fastcall **)(__int64, _BYTE **))(*(_QWORD *)v4 + 144LL))(v4, &v54);
  if ( (_BYTE)v6 )
  {
    v15 = *(_QWORD *)(a1 + 8);
    v72[0] = (__int64)"expected identifier after '.section' directive";
    v73[0] = 259;
    return (unsigned int)sub_3909790(v15, v5, v72, 0, 0);
  }
  v7 = v6;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
  {
    v10 = *(_QWORD *)(a1 + 8);
    v72[0] = (__int64)"unexpected token in '.section' directive";
    v73[0] = 259;
    return (unsigned int)sub_3909CF0(v10, v72, 0, 0, v8, v9);
  }
  v12 = v54;
  if ( !v54 )
  {
    LOBYTE(v69[0]) = 0;
    v67 = (char *)v69;
    v68 = 0;
    goto LABEL_13;
  }
  v13 = (char *)n;
  v67 = (char *)v69;
  v72[0] = n;
  if ( n > 0xF )
  {
    srcb = v54;
    v41 = (char *)sub_22409D0((__int64)&v67, (unsigned __int64 *)v72, 0);
    v12 = srcb;
    v67 = v41;
    v42 = v41;
    v69[0] = v72[0];
  }
  else
  {
    if ( n == 1 )
    {
      LOBYTE(v69[0]) = *v54;
      v14 = (char *)v69;
      goto LABEL_9;
    }
    if ( !n )
    {
      v14 = (char *)v69;
      goto LABEL_9;
    }
    v42 = (char *)v69;
  }
  memcpy(v42, v12, (size_t)v13);
  v13 = (char *)v72[0];
  v14 = v67;
LABEL_9:
  v68 = v13;
  v13[(_QWORD)v14] = 0;
  if ( v68 == (char *)0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
LABEL_13:
  sub_2241490((unsigned __int64 *)&v67, ",", 1u);
  v16 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v17 = (_BYTE *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v16 + 24LL))(v16);
  sub_2241130((unsigned __int64 *)&v67, (size_t)v68, 0, v17, v18);
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v56 = 0;
    v57 = 0;
    v58 = 0;
    v59 = 0;
    sub_38DA070(v70, v67, v68, (unsigned __int64 *)&v56, (unsigned __int64 *)&v58, &v53, &v51, &v52);
    if ( v70[1] )
    {
      v43 = *(_QWORD *)(a1 + 8);
      v73[0] = 260;
      v72[0] = (__int64)v70;
      v7 = sub_3909790(v43, v5, v72, 0, 0);
LABEL_34:
      if ( (__int64 *)v70[0] != &v71 )
        j_j___libc_free_0(v70[0]);
      goto LABEL_15;
    }
    v22 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8)) + 32);
    v72[0] = (__int64)v73;
    srca = (_QWORD *)v22;
    sub_3901E80(v72, *(_BYTE **)(v22 + 696), *(_QWORD *)(v22 + 696) + *(_QWORD *)(v22 + 704));
    v74 = srca[91];
    v75 = srca[92];
    v76 = srca[93];
    if ( (unsigned int)(v74 - 16) > 1 )
    {
      if ( v59 == 13 )
      {
        if ( *(_QWORD *)v58 == 0x6F63747865745F5FLL && v58[2] == 1851747425 && *((_BYTE *)v58 + 12) == 116 )
        {
          v23 = "__text";
          v24 = 6;
          goto LABEL_24;
        }
        if ( *(_QWORD *)v58 == 0x6F63617461645F5FLL && v58[2] == 1851747425 && *((_BYTE *)v58 + 12) == 116 )
        {
          v23 = "__data";
          v24 = 6;
          goto LABEL_24;
        }
      }
      else if ( v59 == 12 && *(_QWORD *)v58 == 0x5F74736E6F635F5FLL && v58[2] == 1818324835 )
      {
        v23 = "__const";
        v24 = 7;
LABEL_24:
        v60[0] = v23;
        v60[1] = v24;
        if ( v5 && (v25 = strlen(v5)) != 0 )
        {
          src = (void *)v25;
          v26 = memchr(v5, 44, v25);
          v27 = (size_t)src;
          if ( !v26 )
          {
            v29 = v5;
LABEL_50:
            srcc = v29;
            v44 = (char *)memchr(v29, 44, v27);
            v29 = srcc;
            v30 = v44;
            if ( !v44 )
              v30 = v5 - 1;
            goto LABEL_29;
          }
          v28 = v26 - v5 + 1;
          v29 = &v5[v28];
          if ( (unsigned __int64)src >= v28 && (unsigned __int64)src > v28 )
          {
            v27 = (size_t)src - v28;
            if ( (__int64)((__int64)src - v28) < 0 )
              v27 = 0x7FFFFFFFFFFFFFFFLL;
            goto LABEL_50;
          }
          v30 = v5 - 1;
        }
        else
        {
          v30 = v5 - 1;
          v29 = v5;
        }
LABEL_29:
        v31 = *(_QWORD *)(a1 + 8);
        v45 = v29;
        v46 = v30;
        v32 = *(void (__fastcall **)(__int64, char *))(*(_QWORD *)v31 + 120LL);
        v63 = 1283;
        v61 = "section \"";
        v62 = &v58;
        v65 = "\" is deprecated";
        v66 = 770;
        v64 = &v61;
        v32(v31, v5);
        v33 = *(_QWORD *)(a1 + 8);
        v34 = *(void (__fastcall **)(__int64, char *, char ***, char *, char *))(*(_QWORD *)v33 + 112LL);
        v61 = "change section name to \"";
        v63 = 1283;
        v62 = v60;
        v65 = "\"";
        v64 = &v61;
        v66 = 770;
        v34(v33, v5, &v64, v45, v46);
      }
    }
    v35 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 8) + 56LL);
    if ( v57 == 6 && *(_DWORD *)v56 == 1163157343 && v56[2] == 21592 )
    {
      v36 = v35();
      v37 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v36 + 160LL);
      v38 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v39 = 1;
    }
    else
    {
      v36 = v35();
      v37 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v36 + 160LL);
      v38 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v39 = 17;
    }
    v40 = sub_38BFA90(v38, v56, v57, v58, v59, v53, v52, v39, 0);
    v37(v36, v40, 0);
    if ( (_WORD *)v72[0] != v73 )
      j_j___libc_free_0(v72[0]);
    goto LABEL_34;
  }
  v21 = *(_QWORD *)(a1 + 8);
  v72[0] = (__int64)"unexpected token in '.section' directive";
  v73[0] = 259;
  v7 = sub_3909CF0(v21, v72, 0, 0, v19, v20);
LABEL_15:
  if ( v67 != (char *)v69 )
    j_j___libc_free_0((unsigned __int64)v67);
  return v7;
}
