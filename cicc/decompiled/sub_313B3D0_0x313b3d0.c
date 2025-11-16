// Function: sub_313B3D0
// Address: 0x313b3d0
//
void __fastcall sub_313B3D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // rdx
  __int64 v5; // r8
  __int64 *v6; // rdi
  unsigned __int64 v7; // r15
  _BYTE *v8; // rax
  __int64 v9; // r12
  int v10; // r9d
  _BYTE *v11; // rcx
  _BYTE *i; // rcx
  _QWORD *v13; // rdi
  __int64 v14; // rdx
  unsigned __int64 v15; // r14
  _BYTE *v16; // r15
  __int64 v17; // rsi
  _BYTE *v18; // rdx
  __int64 v19; // r14
  int v20; // eax
  int v21; // r8d
  _BYTE *v22; // rsi
  _BYTE *v23; // rbx
  unsigned __int64 v24; // rdi
  _BYTE *v25; // r9
  size_t v26; // r8
  _QWORD *v27; // rax
  _BYTE *v28; // r9
  size_t v29; // r8
  _QWORD *v30; // rax
  __int64 v31; // rax
  _QWORD *v32; // rdi
  __int64 v33; // r14
  __int64 v34; // rax
  int v35; // eax
  __int64 v36; // rax
  char v37; // al
  _QWORD *v38; // rax
  _BYTE *v39; // r8
  size_t v40; // r14
  int v41; // eax
  __int64 v42; // rax
  unsigned __int64 *v43; // r14
  int *v44; // rsi
  __int64 v45; // rax
  bool v46; // zf
  _QWORD *v47; // [rsp-10h] [rbp-580h]
  __int64 v48; // [rsp-8h] [rbp-578h]
  size_t n; // [rsp+10h] [rbp-560h]
  size_t na; // [rsp+10h] [rbp-560h]
  void *src; // [rsp+38h] [rbp-538h]
  _BYTE *srca; // [rsp+38h] [rbp-538h]
  void *srcb; // [rsp+38h] [rbp-538h]
  int srcc; // [rsp+38h] [rbp-538h]
  _BYTE *srcd; // [rsp+38h] [rbp-538h]
  _BYTE *v57; // [rsp+68h] [rbp-508h]
  int v58; // [rsp+74h] [rbp-4FCh] BYREF
  int v59; // [rsp+78h] [rbp-4F8h] BYREF
  int v60; // [rsp+7Ch] [rbp-4F4h] BYREF
  __int64 v61; // [rsp+80h] [rbp-4F0h] BYREF
  __int64 v62; // [rsp+88h] [rbp-4E8h] BYREF
  __int64 v63; // [rsp+90h] [rbp-4E0h] BYREF
  __int64 v64; // [rsp+98h] [rbp-4D8h]
  _BYTE **v65; // [rsp+A0h] [rbp-4D0h]
  __int64 *v66; // [rsp+A8h] [rbp-4C8h]
  __int64 *v67; // [rsp+B0h] [rbp-4C0h]
  _QWORD v68[6]; // [rsp+C0h] [rbp-4B0h] BYREF
  _QWORD *v69; // [rsp+F0h] [rbp-480h] BYREF
  size_t v70; // [rsp+F8h] [rbp-478h]
  _QWORD v71[2]; // [rsp+100h] [rbp-470h] BYREF
  __int64 v72; // [rsp+110h] [rbp-460h]
  __int64 v73; // [rsp+118h] [rbp-458h]
  unsigned __int64 v74[2]; // [rsp+120h] [rbp-450h] BYREF
  _QWORD v75[4]; // [rsp+130h] [rbp-440h] BYREF
  _QWORD v76[2]; // [rsp+150h] [rbp-420h] BYREF
  _QWORD v77[2]; // [rsp+160h] [rbp-410h] BYREF
  int v78; // [rsp+170h] [rbp-400h]
  int v79; // [rsp+174h] [rbp-3FCh]
  int v80; // [rsp+178h] [rbp-3F8h]
  int v81; // [rsp+17Ch] [rbp-3F4h]
  char *v82; // [rsp+180h] [rbp-3F0h] BYREF
  __int64 *v83; // [rsp+188h] [rbp-3E8h]
  char v84; // [rsp+190h] [rbp-3E0h] BYREF
  __int64 v85; // [rsp+1A0h] [rbp-3D0h]
  __int64 v86; // [rsp+1A8h] [rbp-3C8h]
  _BYTE *v87; // [rsp+1B0h] [rbp-3C0h] BYREF
  __int64 v88; // [rsp+1B8h] [rbp-3B8h]
  _BYTE v89[944]; // [rsp+1C0h] [rbp-3B0h] BYREF

  v2 = a1 + 712;
  if ( sub_3136D60(a1 + 712) )
    return;
  v6 = *(__int64 **)(a1 + 504);
  v7 = *(unsigned int *)(a1 + 720);
  v8 = v89;
  v9 = *v6;
  v10 = *(_DWORD *)(a1 + 720);
  v87 = v89;
  v88 = 0x1000000000LL;
  if ( v7 )
  {
    v11 = v89;
    if ( v7 > 0x10 )
    {
      sub_313B260((__int64)&v87, v7, v4, (__int64)v89, v5, v7);
      v11 = v87;
      v10 = v7;
      v8 = &v87[56 * (unsigned int)v88];
    }
    for ( i = &v11[56 * v7]; i != v8; v8 += 56 )
    {
      if ( v8 )
      {
        *(_QWORD *)v8 = 0;
        *((_QWORD *)v8 + 1) = v8 + 24;
        *((_QWORD *)v8 + 2) = 0;
        v8[24] = 0;
        *((_DWORD *)v8 + 10) = 0;
        *((_DWORD *)v8 + 11) = 0;
        *((_DWORD *)v8 + 12) = 0;
        *((_DWORD *)v8 + 13) = 0;
      }
    }
    LODWORD(v88) = v10;
    v6 = *(__int64 **)(a1 + 504);
  }
  v62 = v9;
  v61 = a1;
  v64 = sub_BA8E40((__int64)v6, "omp_offload.info", 0x10u);
  src = (void *)v64;
  v83 = &v63;
  v63 = v9;
  v65 = &v87;
  v66 = &v61;
  v67 = &v62;
  v82 = (char *)sub_3122040;
  sub_3136D80(v2, (__int64)&v82);
  v13 = (_QWORD *)v2;
  v68[0] = v9;
  v68[4] = src;
  v82 = (char *)sub_3121CE0;
  v68[1] = &v87;
  v68[3] = &v62;
  v83 = v68;
  v68[2] = &v61;
  sub_3136DD0(v2, (__int64)&v82);
  v14 = (unsigned int)v88;
  v15 = (unsigned __int64)v87;
  if ( v87 == &v87[56 * (unsigned int)v88] )
  {
    if ( *(_QWORD *)(a1 + 496) )
      goto LABEL_27;
    goto LABEL_34;
  }
  v16 = v87;
  v57 = &v87[56 * (unsigned int)v88];
  do
  {
    while ( 1 )
    {
      v19 = *(_QWORD *)v16;
      v20 = *(_DWORD *)(*(_QWORD *)v16 + 32LL);
      if ( !v20 )
      {
        v17 = *(_QWORD *)(v19 + 40);
        if ( !v17 || (v18 = *(_BYTE **)(v19 + 16)) == 0 )
        {
          v69 = v71;
          v25 = (_BYTE *)*((_QWORD *)v16 + 1);
          v26 = *((_QWORD *)v16 + 2);
          if ( &v25[v26] && !v25 )
LABEL_77:
            sub_426248((__int64)"basic_string::_M_construct null not valid");
          v82 = (char *)*((_QWORD *)v16 + 2);
          if ( v26 > 0xF )
          {
            n = v26;
            srca = v25;
            v31 = sub_22409D0((__int64)&v69, (unsigned __int64 *)&v82, 0);
            v25 = srca;
            v26 = n;
            v69 = (_QWORD *)v31;
            v32 = (_QWORD *)v31;
            v71[0] = v82;
          }
          else
          {
            if ( v26 == 1 )
            {
              LOBYTE(v71[0]) = *v25;
              v27 = v71;
              goto LABEL_41;
            }
            if ( !v26 )
            {
              v27 = v71;
              goto LABEL_41;
            }
            v32 = v71;
          }
          memcpy(v32, v25, v26);
          v26 = (size_t)v82;
          v27 = v69;
LABEL_41:
          v70 = v26;
          *((_BYTE *)v27 + v26) = 0;
          v13 = *(_QWORD **)(a1 + 504);
          v22 = v69;
          v72 = *((_QWORD *)v16 + 5);
          v73 = *((_QWORD *)v16 + 6);
          if ( sub_BA8B30((__int64)v13, (__int64)v69, v70) )
          {
            v28 = v69;
            v29 = v70;
            v74[0] = (unsigned __int64)v75;
            if ( (_QWORD *)((char *)v69 + v70) && !v69 )
              goto LABEL_77;
            v82 = (char *)v70;
            if ( v70 > 0xF )
            {
              na = (size_t)v69;
              srcb = (void *)v70;
              v36 = sub_22409D0((__int64)v74, (unsigned __int64 *)&v82, 0);
              v29 = (size_t)srcb;
              v28 = (_BYTE *)na;
              v74[0] = v36;
              v13 = (_QWORD *)v36;
              v75[0] = v82;
            }
            else
            {
              if ( v70 == 1 )
              {
                LOBYTE(v75[0]) = *(_BYTE *)v69;
                v30 = v75;
                goto LABEL_47;
              }
              if ( !v70 )
              {
                v30 = v75;
                goto LABEL_47;
              }
              v13 = v75;
            }
            v22 = v28;
            memcpy(v13, v28, v29);
            v29 = (size_t)v82;
            v30 = (_QWORD *)v74[0];
LABEL_47:
            v74[1] = v29;
            *((_BYTE *)v30 + v29) = 0;
            v58 = 0;
            v75[2] = v72;
            v75[3] = v73;
            if ( !*(_QWORD *)(a2 + 16) )
              goto LABEL_91;
            (*(void (__fastcall **)(__int64, int *, unsigned __int64 *))(a2 + 24))(a2, &v58, v74);
            if ( (_QWORD *)v74[0] != v75 )
              j_j___libc_free_0(v74[0]);
          }
          v13 = v69;
          if ( v69 != v71 )
            j_j___libc_free_0((unsigned __int64)v69);
          goto LABEL_16;
        }
        sub_3136C40(a1, v17, v18, 0, *(_DWORD *)(v19 + 24), 4, (unsigned __int64)byte_3F871B3, 0);
        v13 = v47;
        goto LABEL_16;
      }
      if ( v20 != 1 )
        BUG();
      v21 = *(_DWORD *)(v19 + 24);
      if ( v21 != 1 )
        break;
      if ( !*(_BYTE *)(a1 + 336) )
      {
        v22 = *(_BYTE **)(v19 + 16);
        if ( !v22 )
        {
          v83 = 0;
          v84 = 0;
          v82 = &v84;
          v42 = a2;
          v85 = 0;
          v86 = 0;
          v46 = *(_QWORD *)(a2 + 16) == 0;
          v60 = 2;
          if ( v46 )
LABEL_91:
            sub_4263D6(v13, v22, v14);
          v43 = (unsigned __int64 *)&v82;
          v44 = &v60;
          goto LABEL_83;
        }
        if ( *v22 > 3u )
        {
LABEL_62:
          v13 = (_QWORD *)a1;
          sub_3136C40(
            a1,
            (int)v22,
            v22,
            *(_QWORD *)(v19 + 40),
            v21,
            *(unsigned int *)(v19 + 48),
            (unsigned __int64)byte_3F871B3,
            0);
          v14 = v48;
          goto LABEL_16;
        }
        if ( (v22[32] & 0xFu) - 7 > 1 )
          goto LABEL_60;
      }
LABEL_16:
      v16 += 56;
      if ( v57 == v16 )
        goto LABEL_26;
    }
    if ( (v21 & 0xFFFFFFFD) != 0 )
    {
      v22 = *(_BYTE **)(v19 + 16);
    }
    else
    {
      if ( *(_BYTE *)(a1 + 336) )
      {
        v13 = (_QWORD *)(a1 + 336);
        srcc = *(_DWORD *)(v19 + 24);
        v37 = sub_3122A20(a1 + 336);
        v21 = srcc;
        if ( v37 )
          goto LABEL_16;
      }
      v22 = *(_BYTE **)(v19 + 16);
      if ( !v22 )
      {
        v38 = v77;
        v76[0] = v77;
        v39 = (_BYTE *)*((_QWORD *)v16 + 1);
        v40 = *((_QWORD *)v16 + 2);
        if ( &v39[v40] && !v39 )
          goto LABEL_77;
        v82 = (char *)*((_QWORD *)v16 + 2);
        if ( v40 > 0xF )
        {
          srcd = v39;
          v45 = sub_22409D0((__int64)v76, (unsigned __int64 *)&v82, 0);
          v39 = srcd;
          v76[0] = v45;
          v13 = (_QWORD *)v45;
          v77[0] = v82;
          goto LABEL_85;
        }
        if ( v40 == 1 )
        {
          v14 = (unsigned __int8)*v39;
          LOBYTE(v77[0]) = *v39;
          goto LABEL_81;
        }
        if ( v40 )
        {
          v13 = v77;
LABEL_85:
          v22 = v39;
          memcpy(v13, v39, v40);
          v40 = (size_t)v82;
          v38 = (_QWORD *)v76[0];
        }
LABEL_81:
        v76[1] = v40;
        *((_BYTE *)v38 + v40) = 0;
        v78 = *((_DWORD *)v16 + 10);
        v79 = *((_DWORD *)v16 + 11);
        v80 = *((_DWORD *)v16 + 12);
        v41 = *((_DWORD *)v16 + 13);
        v59 = 1;
        v81 = v41;
        v42 = a2;
        if ( !*(_QWORD *)(a2 + 16) )
          goto LABEL_91;
        v43 = v76;
        v44 = &v59;
LABEL_83:
        (*(void (__fastcall **)(__int64, int *, unsigned __int64 *))(v42 + 24))(v42, v44, v43);
        v13 = v43;
        sub_2240A30(v43);
        goto LABEL_16;
      }
      if ( !*(_QWORD *)(v19 + 40) )
        goto LABEL_16;
    }
    if ( *v22 > 3u )
      goto LABEL_61;
    if ( (v22[32] & 0xFu) - 7 <= 1 )
      goto LABEL_24;
LABEL_60:
    if ( (v22[32] & 0x30) == 0x10 )
    {
LABEL_24:
      if ( v21 == 8 )
        goto LABEL_25;
      goto LABEL_16;
    }
LABEL_61:
    if ( v21 != 8 )
      goto LABEL_62;
LABEL_25:
    v13 = (_QWORD *)a1;
    v16 += 56;
    sub_3136C40(
      a1,
      (int)v22,
      v22,
      *(_QWORD *)(v19 + 40),
      8,
      *(unsigned int *)(v19 + 48),
      *(_QWORD *)(v19 + 56),
      *(_QWORD *)(v19 + 64));
  }
  while ( v57 != v16 );
LABEL_26:
  if ( *(_QWORD *)(a1 + 496) )
  {
LABEL_27:
    if ( !*(_BYTE *)(a1 + 336) )
    {
      v33 = sub_3122A30(a1 + 336);
      v34 = sub_BCE3C0(**(__int64 ***)(a1 + 504), 0);
      v35 = sub_AD6530(v34, 0);
      sub_3717A20(
        *(_QWORD *)(a1 + 504),
        1,
        v35,
        (unsigned int)".requires",
        9,
        0,
        16,
        v33,
        0,
        (__int64)"llvm_offload_entries",
        20);
    }
  }
  v23 = v87;
  v15 = (unsigned __int64)&v87[56 * (unsigned int)v88];
  if ( v87 != (_BYTE *)v15 )
  {
    do
    {
      v15 -= 56LL;
      v24 = *(_QWORD *)(v15 + 8);
      if ( v24 != v15 + 24 )
        j_j___libc_free_0(v24);
    }
    while ( v23 != (_BYTE *)v15 );
    v15 = (unsigned __int64)v87;
  }
LABEL_34:
  if ( (_BYTE *)v15 != v89 )
    _libc_free(v15);
}
