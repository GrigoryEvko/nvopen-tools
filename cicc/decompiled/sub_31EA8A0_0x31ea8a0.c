// Function: sub_31EA8A0
// Address: 0x31ea8a0
//
char __fastcall sub_31EA8A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  void (*v4)(); // rax
  __int64 v5; // r13
  bool v6; // al
  __int64 v7; // rdx
  char v8; // al
  __int64 v9; // rdi
  __int64 *v10; // rax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rsi
  char v14; // r14
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 *v18; // rax
  __int64 *v19; // r15
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // r15
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdi
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // r9
  __int64 v33; // r13
  __int64 v34; // rsi
  __int64 v35; // rdi
  _BYTE *v36; // rax
  __int64 v37; // r15
  int v38; // ecx
  __int64 v39; // rdi
  __int64 *v40; // rdi
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 (__fastcall *v43)(__int64 *, __int64, __int64, __int64); // rax
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rdi
  void *v47; // rax
  __int64 v48; // rdi
  _QWORD *v49; // rax
  __int64 v50; // rdx
  const char *v51; // rax
  __int64 v52; // r14
  __int64 v53; // rsi
  void (*v54)(void); // rax
  __int64 v55; // rax
  __int64 v56; // rsi
  unsigned int v57; // eax
  __int64 v58; // r13
  unsigned int v59; // r12d
  int v60; // edx
  int v61; // ecx
  int v62; // r8d
  int v63; // r9d
  __int64 v64; // rax
  __int64 v65; // r12
  void (__fastcall *v66)(__int64, __int64, unsigned __int64); // r15
  unsigned __int64 v67; // rax
  unsigned int v68; // r10d
  __int64 v69; // rax
  __int64 v71; // [rsp+0h] [rbp-C0h]
  unsigned __int8 v72; // [rsp+13h] [rbp-ADh]
  unsigned int v73; // [rsp+14h] [rbp-ACh]
  __int64 *v74; // [rsp+18h] [rbp-A8h]
  __int64 v75; // [rsp+18h] [rbp-A8h]
  _QWORD v76[4]; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v77; // [rsp+40h] [rbp-80h]
  unsigned __int64 v78; // [rsp+50h] [rbp-70h] BYREF
  __int64 v79; // [rsp+58h] [rbp-68h]
  const char *v80; // [rsp+60h] [rbp-60h] BYREF
  __int64 v81; // [rsp+70h] [rbp-50h]
  __int64 v82; // [rsp+78h] [rbp-48h]
  __int64 v83; // [rsp+80h] [rbp-40h]

  v2 = a2;
  LOBYTE(v4) = sub_23CF310(*(_QWORD *)(a1 + 200));
  if ( (_BYTE)v4 && (*(_BYTE *)(a2 + 33) & 0x1C) != 0 )
    return (char)v4;
  if ( !sub_B2FC80(a2) )
  {
    LOBYTE(v4) = sub_31DC540(a1, a2);
    if ( (_BYTE)v4 )
      return (char)v4;
    v27 = sub_31DB510(a1, a2);
    v29 = *(_QWORD *)(a1 + 360);
    v30 = v27;
    v31 = *(unsigned int *)(a1 + 376);
    if ( (_DWORD)v31 )
    {
      a2 = ((_DWORD)v31 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v28 = v29 + 16 * a2;
      v32 = *(_QWORD *)v28;
      if ( v30 == *(_QWORD *)v28 )
      {
LABEL_28:
        v4 = (void (*)())(v29 + 16 * v31);
        if ( (void (*)())v28 != v4 )
          return (char)v4;
      }
      else
      {
        v28 = 1;
        while ( v32 != -4096 )
        {
          v68 = v28 + 1;
          a2 = ((_DWORD)v31 - 1) & (unsigned int)(v28 + a2);
          v28 = v29 + 16LL * (unsigned int)a2;
          v32 = *(_QWORD *)v28;
          if ( v30 == *(_QWORD *)v28 )
            goto LABEL_28;
          v28 = v68;
        }
      }
    }
    if ( *(_BYTE *)(a1 + 488) )
    {
      v33 = *(_QWORD *)(v2 + 40);
      v34 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 224) + 128LL))(
              *(_QWORD *)(a1 + 224),
              a2,
              v28,
              v30);
      sub_A5BF40((unsigned __int8 *)v2, v34, 0, v33);
      v35 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 224) + 128LL))(*(_QWORD *)(a1 + 224));
      v36 = *(_BYTE **)(v35 + 32);
      if ( (unsigned __int64)v36 >= *(_QWORD *)(v35 + 24) )
      {
        sub_CB5D20(v35, 10);
      }
      else
      {
        *(_QWORD *)(v35 + 32) = v36 + 1;
        *v36 = 10;
      }
    }
  }
  v5 = sub_31DB510(a1, v2);
  v6 = sub_B2FC80(v2);
  sub_31DE970(a1, v5, (*(_BYTE *)(v2 + 32) >> 4) & 3, !v6);
  if ( (*(_BYTE *)(v2 + 34) & 1) != 0 && (*(_BYTE *)sub_B31490(v2, v5, v7) & 4) != 0 )
  {
    v37 = *(_QWORD *)(a1 + 200);
    v78 = (unsigned __int64)&v80;
    sub_31D5230((__int64 *)&v78, *(_BYTE **)(v37 + 512), *(_QWORD *)(v37 + 512) + *(_QWORD *)(v37 + 520));
    v38 = *(_DWORD *)(v37 + 560);
    v81 = *(_QWORD *)(v37 + 544);
    v82 = *(_QWORD *)(v37 + 552);
    v83 = *(_QWORD *)(v37 + 560);
    if ( (_DWORD)v81 != 3 || v38 != 17 )
    {
      v39 = *(_QWORD *)(a1 + 216);
      v76[0] = "tagged symbols (-fsanitize=memtag-globals) are only supported on AArch64 Android";
      v77 = 259;
      sub_E66880(v39, 0, (__int64)v76);
    }
    (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(*(_QWORD *)(a1 + 224), v5, 29);
    if ( (const char **)v78 != &v80 )
      j_j___libc_free_0(v78);
  }
  LOBYTE(v4) = sub_B2FC80(v2);
  if ( (_BYTE)v4 )
    return (char)v4;
  v8 = *(_BYTE *)(v5 + 8);
  if ( (v8 & 4) != 0 )
  {
    if ( (*(_BYTE *)(v5 + 9) & 0x70) == 0x20 )
    {
      *(_WORD *)(v5 + 8) &= 0x8FFBu;
      *(_QWORD *)(v5 + 24) = 0;
      *(_QWORD *)v5 = 0;
    }
    else
    {
      *(_QWORD *)v5 = 0;
      *(_BYTE *)(v5 + 8) = v8 & 0xFB;
    }
  }
  else
  {
    if ( !*(_QWORD *)v5 )
    {
      if ( (*(_BYTE *)(v5 + 9) & 0x70) != 0x20 )
        goto LABEL_11;
      if ( v8 >= 0 )
      {
        v46 = *(_QWORD *)(v5 + 24);
        *(_BYTE *)(v5 + 8) = v8 | 8;
        v47 = sub_E807D0(v46);
        *(_QWORD *)v5 = v47;
        if ( !v47 && (*(_BYTE *)(v5 + 9) & 0x70) != 0x20 )
          goto LABEL_11;
        v8 = *(_BYTE *)(v5 + 8);
      }
    }
    v9 = *(_QWORD *)(a1 + 216);
    if ( (v8 & 1) != 0 )
    {
      v10 = *(__int64 **)(v5 - 8);
      v11 = *v10;
      v12 = v10 + 3;
    }
    else
    {
      v11 = 0;
      v12 = 0;
    }
    v76[2] = v12;
    v78 = (unsigned __int64)v76;
    v80 = "' is already defined";
    v76[0] = "symbol '";
    v76[3] = v11;
    LOWORD(v81) = 770;
    v77 = 1283;
    sub_E66880(v9, 0, (__int64)&v78);
  }
LABEL_11:
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 289LL) )
    (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(*(_QWORD *)(a1 + 224), v5, 4);
  v73 = sub_31578C0(v2, *(_QWORD *)(a1 + 200));
  v71 = sub_B2F730(v2);
  v13 = *(_QWORD *)(v2 + 24);
  v14 = sub_AE5020(v71, v13);
  v15 = sub_9208B0(v71, v13);
  v79 = v16;
  v78 = ((1LL << v14) + ((unsigned __int64)(v15 + 7) >> 3) - 1) >> v14 << v14;
  v17 = sub_CA1930(&v78);
  v72 = sub_31DA250(v2, v71, 0);
  v18 = *(__int64 **)(a1 + 576);
  v19 = v18;
  v74 = &v18[*(unsigned int *)(a1 + 584)];
  if ( v18 != v74 )
  {
    do
    {
      v20 = *v19++;
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v20 + 72LL))(v20, v5, v17);
    }
    while ( v74 != v19 );
  }
  if ( (_BYTE)v73 == 18 )
  {
    v40 = *(__int64 **)(a1 + 224);
    v41 = 1;
    v42 = v72;
    if ( v17 )
      v41 = v17;
    v43 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64))(*v40 + 480);
    goto LABEL_43;
  }
  v21 = sub_31DA6B0(a1);
  v22 = sub_3157C30(v21, v2, v73, *(_QWORD *)(a1 + 200));
  if ( (unsigned __int8)(v73 - 15) <= 2u )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 18LL) && (*(_BYTE *)(v22 + 48) & 0x20) != 0 )
    {
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 560LL))(a1, v2, v5);
      v44 = 1;
      if ( v17 )
        v44 = v17;
      LOBYTE(v4) = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 224)
                                                                                                + 496LL))(
                     *(_QWORD *)(a1 + 224),
                     v22,
                     v5,
                     v44,
                     v72,
                     0);
      return (char)v4;
    }
    if ( (_BYTE)v73 != 16 || v22 != *(_QWORD *)(sub_31DA6B0(a1) + 40) )
    {
LABEL_19:
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 176LL))(*(_QWORD *)(a1 + 224), v22, 0);
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 560LL))(a1, v2, v5);
      sub_31DCA70(a1, v72, v2, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 208LL))(*(_QWORD *)(a1 + 224), v5, 0);
      v24 = sub_31DE680(a1, v2, v23);
      if ( v5 != v24 )
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 208LL))(
          *(_QWORD *)(a1 + 224),
          v24,
          0);
      v25 = *(_QWORD *)(v2 - 32);
      v26 = sub_B2F730(v2);
      sub_31EA6F0(a1, v26, v25, 0);
      if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 289LL) )
      {
        v65 = *(_QWORD *)(a1 + 224);
        v66 = *(void (__fastcall **)(__int64, __int64, unsigned __int64))(*(_QWORD *)v65 + 448LL);
        v67 = sub_E81A90(v17, *(_QWORD **)(a1 + 216), 0, 0);
        v66(v65, v5, v67);
      }
      v4 = *(void (**)())(**(_QWORD **)(a1 + 224) + 160LL);
      if ( v4 == nullsub_99 )
        return (char)v4;
LABEL_69:
      LOBYTE(v4) = ((__int64 (*)(void))v4)();
      return (char)v4;
    }
    v40 = *(__int64 **)(a1 + 224);
    if ( !v17 )
      v17 = 1;
    v45 = *v40;
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 208) + 284LL) )
    {
      v43 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64))(v45 + 488);
    }
    else
    {
      (*(void (__fastcall **)(__int64 *, __int64, __int64))(v45 + 296))(v40, v5, 17);
      v40 = *(__int64 **)(a1 + 224);
      v43 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64))(*v40 + 480);
    }
    v42 = v72;
    v41 = v17;
LABEL_43:
    LOBYTE(v4) = v43(v40, v5, v41, v42);
    return (char)v4;
  }
  if ( (unsigned __int8)(v73 - 12) > 2u || !*(_BYTE *)(*(_QWORD *)(a1 + 208) + 18LL) )
    goto LABEL_19;
  v48 = *(_QWORD *)(a1 + 216);
  if ( (*(_BYTE *)(v5 + 8) & 1) != 0 )
  {
    v49 = *(_QWORD **)(v5 - 8);
    v50 = *v49;
    v51 = (const char *)(v49 + 3);
  }
  else
  {
    v50 = 0;
    v51 = 0;
  }
  v78 = (unsigned __int64)v51;
  v80 = "$tlv$init";
  v79 = v50;
  LOWORD(v81) = 773;
  v75 = sub_E6C460(v48, (const char **)&v78);
  if ( (v73 & 0xFD) == 0xC )
  {
    v69 = sub_31DA6B0(a1);
    (*(void (__fastcall **)(_QWORD, _QWORD, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 504LL))(
      *(_QWORD *)(a1 + 224),
      *(_QWORD *)(v69 + 432),
      v75,
      v17,
      v72);
  }
  else
  {
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 176LL))(*(_QWORD *)(a1 + 224), v22, 0);
    sub_31DCA70(a1, v72, v2, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 208LL))(*(_QWORD *)(a1 + 224), v75, 0);
    v52 = *(_QWORD *)(v2 - 32);
    v53 = sub_B2F730(v2);
    sub_31EA6F0(a1, v53, v52, 0);
  }
  v54 = *(void (**)(void))(**(_QWORD **)(a1 + 224) + 160LL);
  if ( v54 != nullsub_99 )
    v54();
  v55 = sub_31DA6B0(a1);
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 224) + 176LL))(
    *(_QWORD *)(a1 + 224),
    *(_QWORD *)(v55 + 416),
    0);
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 560LL))(a1, v2, v5);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 208LL))(*(_QWORD *)(a1 + 224), v5, 0);
  v56 = *(_QWORD *)(v2 + 8);
  v57 = sub_AE43A0(v71, v56);
  LOWORD(v81) = 259;
  v58 = *(_QWORD *)(a1 + 224);
  v59 = v57 >> 3;
  v78 = (unsigned __int64)"_tlv_bootstrap";
  v64 = sub_31DE8D0(a1, v56, v60, v61, v62, v63, (char)"_tlv_bootstrap");
  sub_E9A500(v58, v64, v59, 0);
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 224) + 536LL))(*(_QWORD *)(a1 + 224), 0, v59);
  sub_E9A500(*(_QWORD *)(a1 + 224), v75, v59, 0);
  v4 = *(void (**)())(**(_QWORD **)(a1 + 224) + 160LL);
  if ( v4 != nullsub_99 )
    goto LABEL_69;
  return (char)v4;
}
