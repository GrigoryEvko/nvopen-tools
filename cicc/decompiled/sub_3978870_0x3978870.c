// Function: sub_3978870
// Address: 0x3978870
//
char __fastcall sub_3978870(__int64 a1, __int64 a2)
{
  void (*v2)(); // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdi
  void (*v6)(); // rcx
  __int64 v7; // r9
  _QWORD *v8; // r12
  __int64 v9; // rsi
  __int64 v10; // rdi
  _BYTE *v11; // rax
  __int64 v12; // r14
  bool v13; // al
  __int64 v14; // rcx
  __int64 v15; // rdx
  char v16; // al
  __int64 v17; // r12
  unsigned int v18; // eax
  __int64 v19; // r14
  unsigned __int64 v20; // r13
  __int64 v21; // rdx
  __int64 v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // rbx
  const char *v25; // rdi
  unsigned int v26; // r14d
  size_t v27; // rax
  char *v28; // rdi
  size_t v29; // rax
  __int64 v30; // r15
  size_t v31; // r13
  size_t v32; // r8
  unsigned __int8 *v33; // rsi
  size_t v34; // rdx
  size_t v35; // rax
  __int64 v36; // rax
  __int64 v37; // r13
  __int64 *v38; // r12
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rbx
  __int64 v42; // rcx
  __int64 v43; // r13
  unsigned int v44; // r12d
  __int64 v45; // rdi
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 (__fastcall *v48)(__int64, _BYTE *, __int64, __int64); // rax
  __int64 v49; // r12
  void (__fastcall *v50)(__int64, _BYTE *, unsigned __int64); // rbx
  unsigned __int64 v51; // rax
  __int64 v52; // r12
  __int64 v53; // rax
  __int64 v54; // rdi
  __int64 *v55; // rax
  __m128i v56; // rax
  __int64 v57; // r12
  __int64 *v58; // r13
  __int64 v59; // rsi
  void (*v60)(void); // rax
  __int64 v61; // r14
  __int64 v62; // rax
  unsigned int v63; // eax
  __int64 *v64; // r14
  unsigned int v65; // r13d
  __int64 v66; // rax
  int v67; // ecx
  int v68; // r10d
  __int64 v69; // r10
  __int64 v70; // rax
  __int64 v72; // [rsp+0h] [rbp-140h]
  unsigned int v73; // [rsp+8h] [rbp-138h]
  unsigned int v74; // [rsp+Ch] [rbp-134h]
  _QWORD *v75; // [rsp+18h] [rbp-128h]
  __int64 v76; // [rsp+20h] [rbp-120h]
  int v77[2]; // [rsp+28h] [rbp-118h]
  __int64 v78; // [rsp+38h] [rbp-108h]
  _BYTE *v79; // [rsp+40h] [rbp-100h]
  char *v81; // [rsp+50h] [rbp-F0h]
  double *v82; // [rsp+58h] [rbp-E8h]
  _QWORD v83[2]; // [rsp+60h] [rbp-E0h] BYREF
  __m128i v84; // [rsp+70h] [rbp-D0h] BYREF
  __int16 v85; // [rsp+80h] [rbp-C0h]
  __m128i v86; // [rsp+90h] [rbp-B0h] BYREF
  char v87; // [rsp+A0h] [rbp-A0h]
  char v88; // [rsp+A1h] [rbp-9Fh]
  __m128i v89[2]; // [rsp+B0h] [rbp-90h] BYREF
  __m128i v90; // [rsp+D0h] [rbp-70h] BYREF
  char v91; // [rsp+E0h] [rbp-60h]
  char v92; // [rsp+E1h] [rbp-5Fh]
  __m128i v93; // [rsp+F0h] [rbp-50h] BYREF
  __int16 v94; // [rsp+100h] [rbp-40h]

  v78 = a2;
  LOBYTE(v2) = sub_17006E0(*(_QWORD *)(a1 + 232));
  if ( (_BYTE)v2 && (*(_BYTE *)(a2 + 33) & 0x1C) != 0 )
    return (char)v2;
  if ( !sub_15E4F60(a2) )
  {
    LOBYTE(v2) = sub_39786C0(a1, a2);
    if ( (_BYTE)v2 )
      return (char)v2;
    v3 = sub_396EAF0(a1, a2);
    v4 = *(unsigned int *)(a1 + 344);
    if ( (_DWORD)v4 )
    {
      v5 = *(_QWORD *)(a1 + 328);
      a2 = ((_DWORD)v4 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v6 = (void (*)())(v5 + 16 * a2);
      v7 = *(_QWORD *)v6;
      if ( v3 == *(_QWORD *)v6 )
      {
LABEL_8:
        v2 = (void (*)())(v5 + 16 * v4);
        if ( v6 != v2 )
          return (char)v2;
      }
      else
      {
        v67 = 1;
        while ( v7 != -8 )
        {
          v68 = v67 + 1;
          a2 = ((_DWORD)v4 - 1) & (unsigned int)(v67 + a2);
          v6 = (void (*)())(v5 + 16LL * (unsigned int)a2);
          v7 = *(_QWORD *)v6;
          if ( v3 == *(_QWORD *)v6 )
            goto LABEL_8;
          v67 = v68;
        }
      }
    }
    if ( *(_BYTE *)(a1 + 416) )
    {
      v8 = *(_QWORD **)(v78 + 40);
      v9 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 256) + 112LL))(
             *(_QWORD *)(a1 + 256),
             a2,
             v3);
      sub_15537D0(v78, v9, 0, v8);
      v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 256) + 112LL))(*(_QWORD *)(a1 + 256));
      v11 = *(_BYTE **)(v10 + 24);
      if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 16) )
      {
        sub_16E7DE0(v10, 10);
      }
      else
      {
        *(_QWORD *)(v10 + 24) = v11 + 1;
        *v11 = 10;
      }
    }
  }
  v12 = sub_396EAF0(a1, v78);
  v79 = (_BYTE *)v12;
  v13 = sub_15E4F60(v78);
  sub_39719F0(a1, v12, (*(_BYTE *)(v78 + 32) >> 4) & 3, !v13);
  LOBYTE(v2) = sub_15E4F60(v78);
  if ( (_BYTE)v2 )
    return (char)v2;
  v15 = *(_QWORD *)v12;
  if ( (*(_BYTE *)(v12 + 8) & 2) != 0 )
  {
    v16 = *(_BYTE *)(v12 + 9);
    if ( (v16 & 0xC) == 8 )
    {
      v16 &= 0xF3u;
      *(_QWORD *)(v12 + 24) = 0;
      *(_BYTE *)(v12 + 9) = v16;
    }
    v14 = v12;
    v15 &= 7u;
    *(_BYTE *)(v12 + 8) &= ~2u;
    *(_QWORD *)v12 = v15;
  }
  else
  {
    if ( (v15 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      goto LABEL_23;
    v16 = *(_BYTE *)(v12 + 9);
  }
  if ( (v16 & 0xC) == 8 )
  {
    *(_BYTE *)(v12 + 8) |= 4u;
    v15 = (__int64)sub_38CE440(*(_QWORD *)(v12 + 24));
    *(_QWORD *)v12 = v15 | *(_QWORD *)v12 & 7LL;
    if ( v15 || (*(_BYTE *)(v12 + 9) & 0xC) == 8 )
    {
LABEL_23:
      v92 = 1;
      v90.m128i_i64[0] = (__int64)"' is already defined";
      v91 = 3;
      v88 = 1;
      v83[0] = sub_3913870((_BYTE *)v12);
      v83[1] = v21;
      v85 = 261;
      v84.m128i_i64[0] = (__int64)v83;
      v86.m128i_i64[0] = (__int64)"symbol '";
      v87 = 3;
      sub_14EC200(v89, &v86, &v84);
      sub_14EC200(&v93, v89, &v90);
      sub_16BCFB0((__int64)&v93, 1u);
    }
  }
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 240) + 305LL) )
    (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 256) + 256LL))(*(_QWORD *)(a1 + 256), v12, 3);
  v17 = 1;
  LOBYTE(v18) = sub_394AFB0(v78, *(_QWORD *)(a1 + 232), v15, v14);
  v74 = v18;
  v72 = sub_1632FA0(*(_QWORD *)(v78 + 40));
  v19 = *(_QWORD *)(*(_QWORD *)v78 + 24LL);
  v20 = (unsigned int)sub_15A9FE0(v72, v19);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v19 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v40 = *(_QWORD *)(v19 + 32);
        v19 = *(_QWORD *)(v19 + 24);
        v17 *= v40;
        continue;
      case 1:
        v22 = 16;
        break;
      case 2:
        v22 = 32;
        break;
      case 3:
      case 9:
        v22 = 64;
        break;
      case 4:
        v22 = 80;
        break;
      case 5:
      case 6:
        v22 = 128;
        break;
      case 7:
        v22 = 8 * (unsigned int)sub_15A9520(v72, 0);
        break;
      case 0xB:
        v22 = *(_DWORD *)(v19 + 8) >> 8;
        break;
      case 0xD:
        v22 = 8LL * *(_QWORD *)sub_15A9930(v72, v19);
        break;
      case 0xE:
        v41 = *(_QWORD *)(v19 + 32);
        v22 = 8 * v41 * sub_12BE0A0(v72, *(_QWORD *)(v19 + 24));
        break;
      case 0xF:
        v22 = 8 * (unsigned int)sub_15A9520(v72, *(_DWORD *)(v19 + 8) >> 8);
        break;
    }
    break;
  }
  v76 = v20 * ((v20 + ((unsigned __int64)(v22 * v17 + 7) >> 3) - 1) / v20);
  v73 = sub_396B790(v78, v72, 0);
  v23 = *(_QWORD **)(a1 + 424);
  v75 = &v23[5 * *(unsigned int *)(a1 + 432)];
  if ( v23 != v75 )
  {
    v24 = *(_QWORD **)(a1 + 424);
    do
    {
      v25 = (const char *)v24[4];
      v26 = unk_4F9E388;
      v81 = (char *)v25;
      v27 = 0;
      if ( v25 )
        v27 = strlen(v25);
      v28 = (char *)v24[3];
      v82 = (double *)v27;
      v29 = 0;
      if ( v28 )
        v29 = strlen(v28);
      v30 = v24[2];
      v31 = v29;
      v32 = 0;
      if ( v30 )
        v32 = strlen((const char *)v24[2]);
      v33 = (unsigned __int8 *)v24[1];
      v34 = 0;
      if ( v33 )
      {
        *(_QWORD *)v77 = v32;
        v35 = strlen((const char *)v24[1]);
        v32 = *(_QWORD *)v77;
        v34 = v35;
      }
      sub_16D8B50((__m128i **)&v93, v33, v34, v30, v32, v26, (unsigned __int8 *)v28, v31, v81, v82);
      (*(void (__fastcall **)(_QWORD, _BYTE *, __int64))(*(_QWORD *)*v24 + 16LL))(*v24, v79, v76);
      if ( v93.m128i_i64[0] )
        sub_16D7950(v93.m128i_i64[0]);
      v24 += 5;
    }
    while ( v75 != v24 );
  }
  if ( (_BYTE)v74 == 16 )
  {
    v52 = 1;
    if ( v76 )
      v52 = v76;
    v53 = sub_396DD80(a1);
    v47 = 0;
    if ( *(_BYTE *)(v53 + 8) )
      v47 = (unsigned int)(1 << v73);
    v46 = v52;
    v45 = *(_QWORD *)(a1 + 256);
    v48 = *(__int64 (__fastcall **)(__int64, _BYTE *, __int64, __int64))(*(_QWORD *)v45 + 368LL);
    goto LABEL_68;
  }
  v36 = sub_396DD80(a1);
  v37 = sub_394B210(v36, v78, v74, *(_QWORD *)(a1 + 232));
  if ( (unsigned __int8)(v74 - 13) <= 2u )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 240) + 19LL)
      && (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v37 + 16LL))(v37) )
    {
      sub_396E9D0(a1, (_BYTE *)v78, (__int64)v79);
      v69 = 1;
      if ( v76 )
        v69 = v76;
      LOBYTE(v2) = (*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 256)
                                                                                                + 384LL))(
                     *(_QWORD *)(a1 + 256),
                     v37,
                     v79,
                     v69,
                     (unsigned int)(1 << v73),
                     0);
      return (char)v2;
    }
    if ( (_BYTE)v74 != 14 || v37 != *(_QWORD *)(sub_396DD80(a1) + 48) )
      goto LABEL_45;
    v42 = v76;
    if ( !v76 )
      v42 = 1;
    v43 = v42;
    v44 = 1 << v73;
    if ( !*(_DWORD *)(*(_QWORD *)(a1 + 240) + 300LL) )
    {
      if ( !*(_BYTE *)(sub_396DD80(a1) + 8) )
        v44 = 0;
      (*(void (__fastcall **)(_QWORD, _BYTE *, __int64))(**(_QWORD **)(a1 + 256) + 256LL))(
        *(_QWORD *)(a1 + 256),
        v79,
        13);
      LOBYTE(v2) = (*(__int64 (__fastcall **)(_QWORD, _BYTE *, __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 368LL))(
                     *(_QWORD *)(a1 + 256),
                     v79,
                     v43,
                     v44);
      return (char)v2;
    }
    v45 = *(_QWORD *)(a1 + 256);
    v46 = v42;
    v47 = v44;
    v48 = *(__int64 (__fastcall **)(__int64, _BYTE *, __int64, __int64))(*(_QWORD *)v45 + 376LL);
LABEL_68:
    LOBYTE(v2) = v48(v45, v79, v46, v47);
    return (char)v2;
  }
  if ( (unsigned __int8)(v74 - 11) <= 1u && *(_BYTE *)(*(_QWORD *)(a1 + 240) + 20LL) )
  {
    v54 = *(_QWORD *)(a1 + 248);
    if ( (*v79 & 4) != 0 )
    {
      v55 = (__int64 *)*((_QWORD *)v79 - 1);
      v56.m128i_i64[1] = *v55;
      v56.m128i_i64[0] = (__int64)(v55 + 2);
    }
    else
    {
      v56 = 0u;
    }
    v90 = v56;
    v93.m128i_i64[0] = (__int64)&v90;
    v93.m128i_i64[1] = (__int64)"$tlv$init";
    v94 = 773;
    v57 = sub_38BF510(v54, (__int64)&v93);
    if ( (_BYTE)v74 == 11 )
    {
      v70 = sub_396DD80(a1);
      (*(void (__fastcall **)(_QWORD, _QWORD, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 392LL))(
        *(_QWORD *)(a1 + 256),
        *(_QWORD *)(v70 + 392),
        v57,
        v76,
        (unsigned int)(1 << v73));
    }
    else
    {
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 160LL))(*(_QWORD *)(a1 + 256), v37, 0);
      sub_396F480(a1, v73, v78);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 176LL))(*(_QWORD *)(a1 + 256), v57, 0);
      v58 = *(__int64 **)(v78 - 24);
      v59 = sub_1632FA0(*(_QWORD *)(v78 + 40));
      sub_3976960(a1, v59, v58);
    }
    v60 = *(void (**)(void))(**(_QWORD **)(a1 + 256) + 144LL);
    if ( v60 != nullsub_581 )
      v60();
    v61 = a1;
    v62 = sub_396DD80(a1);
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(v61 + 256) + 160LL))(
      *(_QWORD *)(v61 + 256),
      *(_QWORD *)(v62 + 376),
      0);
    sub_396E9D0(a1, (_BYTE *)v78, (__int64)v79);
    (*(void (__fastcall **)(_QWORD, _BYTE *, _QWORD))(**(_QWORD **)(v61 + 256) + 176LL))(*(_QWORD *)(v61 + 256), v79, 0);
    v63 = sub_15A9570(v72, *(_QWORD *)v78);
    v64 = *(__int64 **)(a1 + 256);
    v65 = v63 >> 3;
    v66 = sub_3970BC0(a1, (__int64)"_tlv_bootstrap", 14);
    sub_38DDC80(v64, v66, v65, 0);
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 256) + 424LL))(*(_QWORD *)(a1 + 256), 0, v65);
    sub_38DDC80(*(__int64 **)(a1 + 256), v57, v65, 0);
    v2 = *(void (**)())(**(_QWORD **)(a1 + 256) + 144LL);
    if ( v2 != nullsub_581 )
      goto LABEL_48;
    return (char)v2;
  }
LABEL_45:
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 160LL))(*(_QWORD *)(a1 + 256), v37, 0);
  sub_396E9D0(a1, (_BYTE *)v78, (__int64)v79);
  sub_396F480(a1, v73, v78);
  (*(void (__fastcall **)(_QWORD, _BYTE *, _QWORD))(**(_QWORD **)(a1 + 256) + 176LL))(*(_QWORD *)(a1 + 256), v79, 0);
  v38 = *(__int64 **)(v78 - 24);
  v39 = sub_1632FA0(*(_QWORD *)(v78 + 40));
  sub_3976960(a1, v39, v38);
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 240) + 305LL) )
  {
    v49 = *(_QWORD *)(a1 + 256);
    v50 = *(void (__fastcall **)(__int64, _BYTE *, unsigned __int64))(*(_QWORD *)v49 + 344LL);
    v51 = sub_38CB470(v76, *(_QWORD *)(a1 + 248));
    v50(v49, v79, v51);
  }
  v2 = *(void (**)())(**(_QWORD **)(a1 + 256) + 144LL);
  if ( v2 != nullsub_581 )
LABEL_48:
    LOBYTE(v2) = ((__int64 (*)(void))v2)();
  return (char)v2;
}
