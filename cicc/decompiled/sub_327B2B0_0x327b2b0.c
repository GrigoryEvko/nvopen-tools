// Function: sub_327B2B0
// Address: 0x327b2b0
//
__int64 __fastcall sub_327B2B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rcx
  __int64 v4; // rax
  unsigned __int16 v5; // bx
  __int64 v6; // rax
  __int64 v7; // rsi
  __int16 *v8; // rax
  __int16 v9; // dx
  __int64 v10; // rdx
  __int64 v11; // rdx
  char v12; // al
  unsigned int v13; // eax
  unsigned int *v14; // r8
  __int64 v15; // r9
  __int16 v16; // ax
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 v20; // rax
  __int64 v21; // rbx
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 *v27; // rdx
  unsigned int v28; // eax
  __int64 v29; // r13
  __int64 v30; // rax
  __int16 v31; // cx
  __int64 v32; // rsi
  unsigned int *v33; // rbx
  unsigned int *v34; // r15
  __int64 v35; // rax
  __int64 v36; // rdx
  char v37; // al
  bool v38; // al
  const __m128i *v39; // r13
  __int64 v40; // rax
  __int16 v41; // dx
  __int64 v42; // rax
  __int64 v43; // rax
  __m128i v44; // xmm0
  unsigned __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  unsigned __int64 v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rdx
  unsigned __int64 v52; // rax
  unsigned int v53; // r15d
  __int64 *v54; // r13
  __int64 v55; // r14
  unsigned __int64 v56; // rbx
  unsigned __int16 v57; // ax
  int v58; // r8d
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rax
  int v63; // edx
  __int128 v64; // [rsp-10h] [rbp-1C0h]
  __int64 v65; // [rsp+10h] [rbp-1A0h]
  __int64 v66; // [rsp+18h] [rbp-198h]
  __m128i v67; // [rsp+20h] [rbp-190h] BYREF
  __int64 v68; // [rsp+30h] [rbp-180h]
  __int16 v69; // [rsp+3Eh] [rbp-172h]
  __int64 v70; // [rsp+40h] [rbp-170h]
  __int64 v71; // [rsp+48h] [rbp-168h]
  __int64 v72; // [rsp+50h] [rbp-160h]
  __int64 v73; // [rsp+58h] [rbp-158h]
  unsigned __int16 v74; // [rsp+60h] [rbp-150h] BYREF
  __int64 v75; // [rsp+68h] [rbp-148h]
  __int64 v76; // [rsp+70h] [rbp-140h] BYREF
  int v77; // [rsp+78h] [rbp-138h]
  unsigned int v78; // [rsp+80h] [rbp-130h] BYREF
  __int64 v79; // [rsp+88h] [rbp-128h]
  __int64 v80; // [rsp+90h] [rbp-120h] BYREF
  __int64 v81; // [rsp+98h] [rbp-118h]
  __int64 v82; // [rsp+A0h] [rbp-110h] BYREF
  char v83; // [rsp+A8h] [rbp-108h]
  __int64 v84; // [rsp+B0h] [rbp-100h] BYREF
  char v85; // [rsp+B8h] [rbp-F8h]
  __int64 v86; // [rsp+C0h] [rbp-F0h]
  __int64 v87; // [rsp+C8h] [rbp-E8h]
  __int64 v88; // [rsp+D0h] [rbp-E0h]
  __int64 v89; // [rsp+D8h] [rbp-D8h]
  __int64 v90; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v91; // [rsp+E8h] [rbp-C8h]
  unsigned int *v92; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v93; // [rsp+F8h] [rbp-B8h]
  _BYTE v94[176]; // [rsp+100h] [rbp-B0h] BYREF

  v3 = *(_QWORD *)(a2 + 16);
  v4 = *(_QWORD *)(**(_QWORD **)(a1 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 40) + 8LL);
  v5 = *(_WORD *)v4;
  v6 = *(_QWORD *)(v4 + 8);
  v74 = v5;
  v75 = v6;
  if ( !v5 )
  {
    if ( !sub_3007100((__int64)&v74) )
    {
      v7 = *(_QWORD *)(a1 + 80);
      v76 = v7;
      if ( !v7 )
        goto LABEL_5;
      goto LABEL_4;
    }
    return 0;
  }
  if ( *(_QWORD *)(v3 + 8LL * v5 + 112) || (unsigned __int16)(v5 - 176) <= 0x34u )
    return 0;
  v7 = *(_QWORD *)(a1 + 80);
  v76 = v7;
  if ( v7 )
  {
LABEL_4:
    sub_B96E90((__int64)&v76, v7, 1);
    v5 = v74;
  }
LABEL_5:
  v77 = *(_DWORD *)(a1 + 72);
  v8 = *(__int16 **)(a1 + 48);
  v9 = *v8;
  v79 = *((_QWORD *)v8 + 1);
  v92 = (unsigned int *)v94;
  LOWORD(v78) = v9;
  v93 = 0x800000000LL;
  if ( v5 )
  {
    if ( v5 == 1 || (unsigned __int16)(v5 - 504) <= 7u )
      goto LABEL_93;
    v62 = 16LL * (v5 - 1);
    v11 = *(_QWORD *)&byte_444C4A0[v62];
    v12 = byte_444C4A0[v62 + 8];
  }
  else
  {
    v86 = sub_3007260((__int64)&v74);
    v87 = v10;
    v11 = v86;
    v12 = v87;
  }
  v90 = v11;
  LOBYTE(v91) = v12;
  v13 = sub_CA1930(&v90);
  switch ( v13 )
  {
    case 1u:
      v16 = 2;
      goto LABEL_23;
    case 2u:
      v16 = 3;
LABEL_23:
      v17 = 0;
      goto LABEL_24;
    case 4u:
      v16 = 4;
      goto LABEL_23;
    case 8u:
      v16 = 5;
      goto LABEL_23;
    case 0x10u:
      v16 = 6;
      goto LABEL_23;
    case 0x20u:
      v16 = 7;
      goto LABEL_23;
    case 0x40u:
      v16 = 8;
      goto LABEL_23;
    case 0x80u:
      v16 = 9;
      goto LABEL_23;
  }
  v16 = sub_3007020(*(_QWORD **)(a2 + 64), v13);
LABEL_24:
  LOWORD(v80) = v16;
  v20 = *(unsigned int *)(a1 + 64);
  v21 = *(_QWORD *)(a1 + 40);
  v81 = v17;
  v68 = v21 + 40 * v20;
  if ( v68 == v21 )
    goto LABEL_70;
  v65 = 0;
  v69 = 0;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v22 = *(_DWORD *)(*(_QWORD *)v21 + 24LL);
        if ( v22 == 234 )
        {
          v39 = *(const __m128i **)(*(_QWORD *)v21 + 40LL);
          v40 = *(_QWORD *)(v39->m128i_i64[0] + 48) + 16LL * v39->m128i_u32[2];
          v41 = *(_WORD *)v40;
          v42 = *(_QWORD *)(v40 + 8);
          LOWORD(v90) = v41;
          v91 = v42;
          if ( v41 )
          {
            if ( (unsigned __int16)(v41 - 17) <= 0xD3u )
              goto LABEL_62;
          }
          else if ( sub_30070B0((__int64)&v90) )
          {
LABEL_62:
            v14 = v92;
            v18 = 0;
            goto LABEL_39;
          }
          v43 = (unsigned int)v93;
          v44 = _mm_loadu_si128(v39);
          v45 = (unsigned int)v93 + 1LL;
          if ( v45 > HIDWORD(v93) )
          {
            v67 = v44;
            sub_C8D5F0((__int64)&v92, v94, v45, 0x10u, (__int64)v14, v15);
            v43 = (unsigned int)v93;
            v44 = _mm_load_si128(&v67);
          }
          *(__m128i *)&v92[4 * v43] = v44;
          v28 = v93 + 1;
          LODWORD(v93) = v93 + 1;
        }
        else
        {
          if ( v22 != 51 )
            goto LABEL_62;
          v23 = sub_33F17F0(a2, 51, &v76, (unsigned int)v80, v81);
          v15 = v24;
          v25 = (unsigned int)v93;
          v26 = v23;
          if ( (unsigned __int64)(unsigned int)v93 + 1 > HIDWORD(v93) )
          {
            v67.m128i_i64[0] = v23;
            v67.m128i_i64[1] = v15;
            sub_C8D5F0((__int64)&v92, v94, (unsigned int)v93 + 1LL, 0x10u, v23, v15);
            v25 = (unsigned int)v93;
            v15 = v67.m128i_i64[1];
            v26 = v67.m128i_i64[0];
          }
          v27 = (__int64 *)&v92[4 * v25];
          *v27 = v26;
          v27[1] = v15;
          v28 = v93 + 1;
          LODWORD(v93) = v93 + 1;
        }
        v14 = v92;
        v29 = 4LL * v28;
        v30 = *(_QWORD *)(*(_QWORD *)&v92[v29 - 4] + 48LL) + 16LL * v92[v29 - 2];
        v31 = *(_WORD *)v30;
        v32 = *(_QWORD *)(v30 + 8);
        LOWORD(v90) = v31;
        v91 = v32;
        if ( !v31 )
          break;
        if ( (unsigned __int16)(v31 - 10) <= 6u
          || (unsigned __int16)(v31 - 126) <= 0x31u
          || (unsigned __int16)(v31 - 208) <= 0x14u )
        {
          v65 = v32;
          v69 = v31;
        }
        else if ( (unsigned __int16)(v31 - 2) > 7u
               && (unsigned __int16)(v31 - 17) > 0x6Cu
               && (unsigned __int16)(v31 - 176) > 0x1Fu )
        {
LABEL_38:
          v18 = 0;
          goto LABEL_39;
        }
        v21 += 40;
        if ( v68 == v21 )
          goto LABEL_45;
      }
      v66 = (__int64)v92;
      v67.m128i_i64[0] = v32;
      v37 = sub_3007030((__int64)&v90);
      v14 = (unsigned int *)v66;
      if ( !v37 )
        break;
      v65 = v67.m128i_i64[0];
      v21 += 40;
      v69 = 0;
      if ( v68 == v21 )
        goto LABEL_45;
    }
    v67.m128i_i64[0] = v66;
    v38 = sub_3007070((__int64)&v90);
    v14 = (unsigned int *)v67.m128i_i64[0];
    if ( !v38 )
      goto LABEL_38;
    v21 += 40;
  }
  while ( v68 != v21 );
LABEL_45:
  if ( v69 || v65 )
  {
    v33 = &v14[v29];
    LOWORD(v80) = v69;
    v81 = v65;
    if ( v14 != &v14[v29] )
    {
      v34 = v14;
      while ( 1 )
      {
        v35 = *(_QWORD *)(*(_QWORD *)v34 + 48LL) + 16LL * v34[2];
        if ( *(_WORD *)v35 != v69 || *(_QWORD *)(v35 + 8) != v65 && !v69 )
        {
          if ( *(_DWORD *)(*(_QWORD *)v34 + 24LL) == 51 )
          {
            v72 = sub_33F17F0(a2, 51, &v76, (unsigned int)v80, v81);
            v73 = v46;
            *(_QWORD *)v34 = v72;
            v34[2] = v73;
          }
          else
          {
            v70 = sub_33FB890(a2, (unsigned int)v80, v81, *(_QWORD *)v34, *((_QWORD *)v34 + 1));
            v71 = v36;
            *(_QWORD *)v34 = v70;
            v34[2] = v71;
          }
        }
        v34 += 4;
        if ( v33 == v34 )
          break;
        v69 = v80;
        v65 = v81;
      }
    }
  }
LABEL_70:
  if ( !(_WORD)v78 )
  {
    v47 = sub_3007260((__int64)&v78);
    v90 = v47;
    v91 = v48;
    goto LABEL_72;
  }
  if ( (_WORD)v78 == 1 || (unsigned __int16)(v78 - 504) <= 7u )
    goto LABEL_93;
  v48 = 16LL * ((unsigned __int16)v78 - 1);
  v47 = *(_QWORD *)&byte_444C4A0[v48];
  LOBYTE(v48) = byte_444C4A0[v48 + 8];
LABEL_72:
  v82 = v47;
  v83 = v48;
  v49 = sub_CA1930(&v82);
  if ( !(_WORD)v80 )
  {
    v50 = sub_3007260((__int64)&v80);
    v88 = v50;
    v89 = v51;
    goto LABEL_74;
  }
  if ( (_WORD)v80 == 1 || (unsigned __int16)(v80 - 504) <= 7u )
LABEL_93:
    BUG();
  v51 = 16LL * ((unsigned __int16)v80 - 1);
  v50 = *(_QWORD *)&byte_444C4A0[v51];
  LOBYTE(v51) = byte_444C4A0[v51 + 8];
LABEL_74:
  v85 = v51;
  v84 = v50;
  v52 = sub_CA1930(&v84);
  v53 = v80;
  v54 = *(__int64 **)(a2 + 64);
  v55 = v81;
  v56 = v49 / v52;
  v57 = sub_2D43050(v80, v56);
  v58 = 0;
  if ( !v57 )
  {
    v57 = sub_3009400(v54, v53, v55, (unsigned int)v56, 0);
    v58 = v63;
  }
  *((_QWORD *)&v64 + 1) = (unsigned int)v93;
  *(_QWORD *)&v64 = v92;
  v59 = sub_33FC220(a2, 156, (unsigned int)&v76, v57, v58, (unsigned int)&v76, v64);
  v61 = sub_33FB890(a2, v78, v79, v59, v60);
  v14 = v92;
  v18 = v61;
LABEL_39:
  if ( v14 != (unsigned int *)v94 )
    _libc_free((unsigned __int64)v14);
  if ( v76 )
    sub_B91220((__int64)&v76, v76);
  return v18;
}
