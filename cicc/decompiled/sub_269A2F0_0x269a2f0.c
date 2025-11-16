// Function: sub_269A2F0
// Address: 0x269a2f0
//
__int64 __fastcall sub_269A2F0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rsi
  unsigned __int8 v5; // al
  __int64 *v6; // rax
  __int64 v7; // r15
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 **v10; // rax
  __int64 *v11; // r14
  __int64 result; // rax
  __int64 *i; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int8 *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdi
  bool v21; // r14
  _BYTE *v22; // r15
  unsigned int v23; // eax
  __int64 v24; // rsi
  __int64 v25; // rax
  unsigned int v26; // edx
  __int64 v27; // rsi
  _QWORD *v28; // rax
  __int64 v29; // r14
  _QWORD *v30; // rax
  __int64 v31; // rax
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 v34; // r9
  unsigned __int8 *v35; // rdi
  _BYTE *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdi
  unsigned __int8 *v40; // rax
  bool v41; // zf
  __int64 v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r15
  __int64 v53; // r15
  __int64 v54; // rdx
  _BYTE *v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rdi
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rdi
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rdi
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rdi
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rdi
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rdi
  __int64 v77; // [rsp+8h] [rbp-138h]
  __int64 v78; // [rsp+8h] [rbp-138h]
  __int64 v79; // [rsp+10h] [rbp-130h]
  __int64 v80; // [rsp+10h] [rbp-130h]
  __int64 v81; // [rsp+10h] [rbp-130h]
  __int64 v82; // [rsp+18h] [rbp-128h]
  __int64 v83; // [rsp+20h] [rbp-120h]
  __int64 v84; // [rsp+20h] [rbp-120h]
  __int64 v85; // [rsp+20h] [rbp-120h]
  __int64 v86; // [rsp+20h] [rbp-120h]
  __int64 v87; // [rsp+20h] [rbp-120h]
  __int64 v88; // [rsp+20h] [rbp-120h]
  char v90; // [rsp+37h] [rbp-109h] BYREF
  __int64 v91; // [rsp+38h] [rbp-108h] BYREF
  _QWORD v92[2]; // [rsp+40h] [rbp-100h] BYREF
  _QWORD v93[2]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 (__fastcall *v94)(const __m128i **, const __m128i *, int); // [rsp+60h] [rbp-E0h]
  __int64 (__fastcall *v95)(__int64 *, __int64, __int64 *, _BYTE *); // [rsp+68h] [rbp-D8h]
  _QWORD v96[2]; // [rsp+70h] [rbp-D0h] BYREF
  __int64 (__fastcall *v97)(const __m128i **, const __m128i *, int); // [rsp+80h] [rbp-C0h]
  __int64 (__fastcall *v98)(__int64 *, __int64, __int64 *); // [rsp+88h] [rbp-B8h]
  _QWORD v99[2]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 (__fastcall *v100)(const __m128i **, const __m128i *, int); // [rsp+A0h] [rbp-A0h]
  __int64 (__fastcall *v101)(__int64 *, __int64, __int64 *); // [rsp+A8h] [rbp-98h]
  _QWORD v102[2]; // [rsp+B0h] [rbp-90h] BYREF
  __int64 (__fastcall *v103)(const __m128i **, const __m128i *, int); // [rsp+C0h] [rbp-80h]
  __int64 (__fastcall *v104)(__int64, __int64, __int64 *); // [rsp+C8h] [rbp-78h]
  _QWORD *v105; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v106; // [rsp+D8h] [rbp-68h]
  _QWORD v107[12]; // [rsp+E0h] [rbp-60h] BYREF

  v3 = *(_QWORD *)(a2 + 208);
  v4 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
  if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
    v4 = *(_QWORD *)(v4 + 24);
  v5 = *(_BYTE *)v4;
  if ( *(_BYTE *)v4 )
  {
    if ( v5 == 22 )
    {
      v4 = *(_QWORD *)(v4 + 24);
    }
    else if ( v5 <= 0x1Cu )
    {
      v4 = 0;
    }
    else
    {
      v4 = sub_B43CB0(v4);
    }
  }
  v91 = v4;
  v105 = v107;
  v106 = 0x800000000LL;
  v6 = sub_267FA80(v3 + 28312, v4);
  v7 = *v6;
  v8 = *v6 + 8LL * *((unsigned int *)v6 + 2);
  if ( *v6 != v8 )
  {
    do
    {
      v9 = *(_QWORD *)(*(_QWORD *)v7 + 24LL);
      if ( *(_BYTE *)v9 != 85 || *(_QWORD *)v7 != v9 - 32 )
        goto LABEL_9;
      if ( *(char *)(v9 + 7) < 0 )
      {
        v82 = *(_QWORD *)(*(_QWORD *)v7 + 24LL);
        v44 = sub_BD2BC0(v82);
        v9 = v82;
        v46 = v45 + v44;
        v85 = v46;
        if ( *(char *)(v82 + 7) >= 0 )
        {
          v48 = v46 >> 4;
        }
        else
        {
          v47 = sub_BD2BC0(v82);
          v9 = v82;
          v48 = (v85 - v47) >> 4;
        }
        if ( (_DWORD)v48 )
          goto LABEL_9;
      }
      v49 = *(_QWORD *)(v3 + 28432);
      if ( !v49
        || (v50 = *(_QWORD *)(v9 - 32)) == 0
        || *(_BYTE *)v50
        || *(_QWORD *)(v50 + 24) != *(_QWORD *)(v9 + 80)
        || v49 != v50 )
      {
LABEL_9:
        v9 = 0;
      }
      v7 += 8;
      *(_QWORD *)(a1 + 296) = v9;
    }
    while ( v8 != v7 );
  }
  v105 = v107;
  v106 = 0x800000000LL;
  v10 = (__int64 **)sub_267FA80(v3 + 28472, v91);
  v11 = *v10;
  result = *((unsigned int *)v10 + 2);
  for ( i = &v11[result]; i != v11; *(_QWORD *)(a1 + 312) = v14 )
  {
    result = *v11;
    v14 = *(_QWORD *)(*v11 + 24);
    if ( *(_BYTE *)v14 != 85 || result != v14 - 32 )
      goto LABEL_13;
    if ( *(char *)(v14 + 7) < 0 )
    {
      v86 = *(_QWORD *)(*v11 + 24);
      result = sub_BD2BC0(v86);
      v14 = v86;
      v52 = result + v51;
      if ( *(char *)(v86 + 7) >= 0 )
      {
        v53 = v52 >> 4;
      }
      else
      {
        result = sub_BD2BC0(v86);
        v14 = v86;
        v53 = (v52 - result) >> 4;
      }
      if ( (_DWORD)v53 )
        goto LABEL_13;
    }
    v54 = *(_QWORD *)(v3 + 28592);
    if ( !v54
      || (result = *(_QWORD *)(v14 - 32)) == 0
      || *(_BYTE *)result
      || *(_QWORD *)(result + 24) != *(_QWORD *)(v14 + 80)
      || v54 != result )
    {
LABEL_13:
      v14 = 0;
    }
    ++v11;
  }
  if ( *(_QWORD *)(a1 + 296) && *(_QWORD *)(a1 + 312) )
  {
    sub_2699F90(a1 + 344, &v91);
    v15 = *(_QWORD *)(a1 + 296);
    *(_BYTE *)(a1 + 320) = 1;
    v16 = sub_2674090(v15, (__int64)&v91);
    v17 = *(_QWORD *)(a1 + 296);
    *(_QWORD *)(a1 + 304) = v16;
    v18 = sub_2674070(v17, (__int64)&v91);
    v93[0] = a1;
    v95 = sub_266E2C0;
    v94 = sub_266E050;
    v93[1] = a2;
    sub_267EC80(a2, (__int64)v18, (__int64)v93);
    v21 = 1;
    if ( *(_BYTE *)(v3 + 34976) )
    {
      v19 = *(_QWORD *)(v3 + 4592);
      if ( !v19 || sub_B2FC80(v19) || (v20 = *(_QWORD *)(v3 + 33552)) == 0 || sub_B2FC80(v20) )
        v21 = 0;
    }
    v22 = sub_2674040(*(unsigned __int8 **)(a1 + 304));
    v23 = *((_DWORD *)v22 + 8);
    if ( v23 > 0x40 )
    {
      v24 = **((_QWORD **)v22 + 3) | 3LL;
    }
    else
    {
      v24 = 3;
      if ( v23 )
        v24 = ((__int64)(*((_QWORD *)v22 + 3) << (64 - (unsigned __int8)v23)) >> (64 - (unsigned __int8)v23)) | 3;
    }
    v25 = sub_ACD640(*((_QWORD *)v22 + 1), v24, 0);
    v26 = *((_DWORD *)v22 + 8);
    v27 = v25;
    v28 = (_QWORD *)*((_QWORD *)v22 + 3);
    if ( v26 > 0x40 )
    {
      if ( (*v28 & 2) == 0 )
        goto LABEL_29;
    }
    else if ( !v26 || (((__int64)((_QWORD)v28 << (64 - (unsigned __int8)v26)) >> (64 - (unsigned __int8)v26)) & 2) == 0 )
    {
LABEL_29:
      if ( byte_4FF4DC8 != 1 && v21 )
      {
        v74 = sub_2673FD0(*(unsigned __int8 **)(a1 + 304));
        LODWORD(v105) = 2;
        v75 = sub_AAAE30(v74, v27, &v105, 1);
        v76 = *(_QWORD *)(a1 + 304);
        LODWORD(v105) = 0;
        *(_QWORD *)(a1 + 304) = sub_AAAE30(v76, v75, &v105, 1);
      }
      else
      {
        *(_BYTE *)(a1 + 241) = *(_BYTE *)(a1 + 240);
      }
      goto LABEL_32;
    }
    *(_BYTE *)(a1 + 240) = *(_BYTE *)(a1 + 241);
LABEL_32:
    v29 = *(_QWORD *)(v91 + 40);
    v105 = v107;
    sub_266F100((__int64 *)&v105, *(_BYTE **)(v29 + 232), *(_QWORD *)(v29 + 232) + *(_QWORD *)(v29 + 240));
    v107[2] = *(_QWORD *)(v29 + 264);
    v107[3] = *(_QWORD *)(v29 + 272);
    v107[4] = *(_QWORD *)(v29 + 280);
    v30 = (_QWORD *)sub_B2BE50(v91);
    v83 = sub_BCB2D0(v30);
    v31 = sub_3135E00(&v105, v91);
    v32 = v31 >> 32;
    if ( (_DWORD)v31 )
    {
      v78 = v31 >> 32;
      v81 = sub_ACD640(v83, (int)v31, 0);
      v70 = sub_2673FD0(*(unsigned __int8 **)(a1 + 304));
      LODWORD(v102[0]) = 3;
      v71 = sub_AAAE30(v70, v81, v102, 1);
      v72 = *(_QWORD *)(a1 + 304);
      LODWORD(v102[0]) = 0;
      v73 = sub_AAAE30(v72, v71, v102, 1);
      LODWORD(v32) = v78;
      *(_QWORD *)(a1 + 304) = v73;
    }
    if ( (_DWORD)v32 )
    {
      v80 = sub_ACD640(v83, (int)v32, 0);
      v67 = sub_2673FD0(*(unsigned __int8 **)(a1 + 304));
      LODWORD(v102[0]) = 4;
      v68 = sub_AAAE30(v67, v80, v102, 1);
      v69 = *(_QWORD *)(a1 + 304);
      LODWORD(v102[0]) = 0;
      *(_QWORD *)(a1 + 304) = sub_AAAE30(v69, v68, v102, 1);
    }
    v33 = sub_3135FE0(&v105, v91);
    v34 = v33 >> 32;
    if ( (_DWORD)v33 )
    {
      v77 = v33 >> 32;
      v79 = sub_ACD640(v83, (int)v33, 0);
      v63 = sub_2673FD0(*(unsigned __int8 **)(a1 + 304));
      LODWORD(v102[0]) = 5;
      v64 = sub_AAAE30(v63, v79, v102, 1);
      v65 = *(_QWORD *)(a1 + 304);
      LODWORD(v102[0]) = 0;
      v66 = sub_AAAE30(v65, v64, v102, 1);
      LODWORD(v34) = v77;
      *(_QWORD *)(a1 + 304) = v66;
    }
    if ( (_DWORD)v34 )
    {
      v88 = sub_ACD640(v83, (int)v34, 0);
      v59 = sub_2673FD0(*(unsigned __int8 **)(a1 + 304));
      LODWORD(v102[0]) = 6;
      v60 = sub_AAAE30(v59, v88, v102, 1);
      v61 = *(_QWORD *)(a1 + 304);
      LODWORD(v102[0]) = 0;
      v62 = sub_AAAE30(v61, v60, v102, 1);
      *(_QWORD *)(a1 + 304) = v62;
      v35 = (unsigned __int8 *)v62;
    }
    else
    {
      v35 = *(unsigned __int8 **)(a1 + 304);
    }
    v36 = sub_2674010(v35);
    v84 = sub_ACD640(*((_QWORD *)v36 + 1), *(unsigned __int8 *)(a1 + 464), 0);
    v37 = sub_2673FD0(*(unsigned __int8 **)(a1 + 304));
    LODWORD(v102[0]) = 1;
    v38 = sub_AAAE30(v37, v84, v102, 1);
    v39 = *(_QWORD *)(a1 + 304);
    LODWORD(v102[0]) = 0;
    v40 = (unsigned __int8 *)sub_AAAE30(v39, v38, v102, 1);
    v41 = (_BYTE)qword_4FF4C08 == 0;
    *(_QWORD *)(a1 + 304) = v40;
    if ( v41 )
    {
      v55 = sub_2673FE0(v40);
      v87 = sub_ACD640(*((_QWORD *)v55 + 1), 0, 0);
      v56 = sub_2673FD0(*(unsigned __int8 **)(a1 + 304));
      LODWORD(v102[0]) = 0;
      v57 = sub_AAAE30(v56, v87, v102, 1);
      v58 = *(_QWORD *)(a1 + 304);
      LODWORD(v102[0]) = 0;
      *(_QWORD *)(a1 + 304) = sub_AAAE30(v58, v57, v102, 1);
    }
    v92[0] = v3;
    v96[0] = a1;
    v92[1] = a2;
    v96[1] = &v90;
    v98 = sub_266EE20;
    v97 = sub_266E080;
    v42 = *(_QWORD *)(a1 + 296);
    v43 = *(_QWORD *)(v42 - 32);
    if ( v43 )
    {
      if ( *(_BYTE *)v43 )
      {
        v43 = 0;
      }
      else if ( *(_QWORD *)(v43 + 24) != *(_QWORD *)(v42 + 80) )
      {
        v43 = 0;
      }
    }
    if ( !sub_B2FC80(v43) )
    {
      sub_267F450(v92, 15, (__int64)v96);
      sub_267F450(v92, 16, (__int64)v96);
      sub_267F450(v92, 188, (__int64)v96);
      sub_267F450(v92, 171, (__int64)v96);
      sub_267F450(v92, 172, (__int64)v96);
    }
    if ( *(_BYTE *)(a1 + 241) == *(_BYTE *)(a1 + 240) )
    {
      sub_A17130((__int64)v96);
      sub_2240A30((unsigned __int64 *)&v105);
      return (__int64)sub_A17130((__int64)v93);
    }
    else
    {
      v101 = sub_266EDE0;
      v100 = sub_266E0B0;
      v99[1] = &v90;
      v99[0] = a1;
      sub_267F450(v92, 6, (__int64)v99);
      v102[0] = a1;
      v102[1] = &v90;
      v104 = sub_266ED50;
      v103 = sub_266E0E0;
      sub_267F450(v92, 187, (__int64)v102);
      if ( v103 )
        v103((const __m128i **)v102, (const __m128i *)v102, 3);
      if ( v100 )
        v100((const __m128i **)v99, (const __m128i *)v99, 3);
      if ( v97 )
        v97((const __m128i **)v96, (const __m128i *)v96, 3);
      if ( v105 != v107 )
        j_j___libc_free_0((unsigned __int64)v105);
      result = (__int64)v94;
      if ( v94 )
        return v94((const __m128i **)v93, (const __m128i *)v93, 3);
    }
  }
  return result;
}
