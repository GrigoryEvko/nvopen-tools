// Function: sub_F3E970
// Address: 0xf3e970
//
__int64 __fastcall sub_F3E970(__int64 a1, __int64 a2)
{
  _QWORD **v2; // r12
  bool v3; // zf
  __int64 *v4; // rax
  __int64 v5; // rdi
  _QWORD *v6; // rdx
  _QWORD *v7; // r14
  __int64 *v8; // r13
  int v9; // r12d
  int v10; // eax
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // rbx
  __int64 v14; // r12
  __m128i *v15; // rdx
  unsigned int v16; // eax
  unsigned __int64 v17; // rbx
  __int64 v18; // rsi
  __int64 v19; // rax
  unsigned __int8 v20; // dl
  __int64 v21; // r12
  __int64 v22; // r13
  _QWORD *v23; // rbx
  _QWORD *v24; // r13
  __int64 v25; // rax
  int v26; // eax
  __int64 v27; // rax
  unsigned __int8 v28; // dl
  __int64 v29; // r15
  _QWORD *v30; // r14
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  _QWORD **v40; // rbx
  int v41; // eax
  _QWORD *v42; // rdi
  _QWORD **v44; // rbx
  _QWORD *v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rax
  bool v53; // al
  _QWORD *v54; // [rsp+10h] [rbp-230h]
  _QWORD *v55; // [rsp+20h] [rbp-220h]
  int v56; // [rsp+3Ch] [rbp-204h]
  __m128i *v57; // [rsp+40h] [rbp-200h]
  __int64 v58; // [rsp+48h] [rbp-1F8h]
  __int64 v59; // [rsp+50h] [rbp-1F0h]
  int v60; // [rsp+58h] [rbp-1E8h]
  unsigned int v61; // [rsp+5Ch] [rbp-1E4h]
  int v62; // [rsp+6Ch] [rbp-1D4h] BYREF
  __int64 v63; // [rsp+70h] [rbp-1D0h] BYREF
  __int64 v64; // [rsp+78h] [rbp-1C8h] BYREF
  __int64 v65; // [rsp+80h] [rbp-1C0h] BYREF
  __int128 v66; // [rsp+88h] [rbp-1B8h] BYREF
  char v67; // [rsp+98h] [rbp-1A8h]
  __int64 v68; // [rsp+A0h] [rbp-1A0h]
  __m128i v69; // [rsp+B0h] [rbp-190h] BYREF
  char v70; // [rsp+C8h] [rbp-178h]
  __int64 v71; // [rsp+D0h] [rbp-170h]
  __int64 v72[4]; // [rsp+E0h] [rbp-160h] BYREF
  __int64 v73; // [rsp+100h] [rbp-140h]
  _BYTE *v74; // [rsp+110h] [rbp-130h] BYREF
  __int64 v75; // [rsp+118h] [rbp-128h]
  _BYTE v76[64]; // [rsp+120h] [rbp-120h] BYREF
  __int64 v77; // [rsp+160h] [rbp-E0h] BYREF
  __int64 v78; // [rsp+168h] [rbp-D8h]
  __int64 *v79; // [rsp+170h] [rbp-D0h] BYREF
  unsigned int v80; // [rsp+178h] [rbp-C8h]
  _BYTE v81[48]; // [rsp+210h] [rbp-30h] BYREF

  v3 = *(_BYTE *)(a1 + 40) == 0;
  v74 = v76;
  v75 = 0x800000000LL;
  v4 = (__int64 *)&v79;
  v77 = 0;
  v78 = 1;
  if ( !v3 )
  {
    do
    {
      *v4 = 0;
      v4 += 5;
      *((_BYTE *)v4 - 16) = 0;
      *(v4 - 1) = 0;
    }
    while ( v4 != (__int64 *)v81 );
    v54 = (_QWORD *)(a1 + 48);
    v55 = (_QWORD *)(*(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (_QWORD *)(a1 + 48) != v55 )
    {
      while ( 1 )
      {
        if ( !v55 )
          BUG();
        v5 = v55[5];
        if ( v5 )
        {
          v59 = sub_B14240(v5);
          v7 = v6;
          if ( (_QWORD *)v59 != v6 )
            break;
        }
LABEL_77:
        sub_F38400((__int64)&v77);
        v55 = (_QWORD *)(*v55 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v54 == v55 )
        {
          v44 = (_QWORD **)v74;
          v2 = (_QWORD **)&v74[8 * (unsigned int)v75];
          v41 = v75;
          if ( v2 != (_QWORD **)v74 )
          {
            do
            {
              v45 = *v44++;
              sub_B14290(v45);
            }
            while ( v2 != v44 );
            goto LABEL_69;
          }
          goto LABEL_70;
        }
      }
      while ( 1 )
      {
        while ( 1 )
        {
          v17 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
          if ( *(_BYTE *)(v17 + 32) != 1 )
          {
            if ( *(_BYTE *)(v17 + 64) )
              break;
          }
          sub_F38400((__int64)&v77);
          v7 = (_QWORD *)(*v7 & 0xFFFFFFFFFFFFFFF8LL);
          if ( v7 == (_QWORD *)v59 )
            goto LABEL_77;
        }
        v18 = *(_QWORD *)(v17 + 24);
        v72[0] = v18;
        if ( v18 )
          sub_B96E90((__int64)v72, v18, 1);
        v19 = sub_B10CD0((__int64)v72);
        v20 = *(_BYTE *)(v19 - 16);
        if ( (v20 & 2) != 0 )
        {
          if ( *(_DWORD *)(v19 - 24) != 2 )
            goto LABEL_29;
          v46 = *(_QWORD *)(v19 - 32);
        }
        else
        {
          if ( ((*(_WORD *)(v19 - 16) >> 6) & 0xF) != 2 )
          {
LABEL_29:
            v21 = 0;
            goto LABEL_30;
          }
          v46 = v19 - 16 - 8LL * ((v20 >> 2) & 0xF);
        }
        v21 = *(_QWORD *)(v46 + 8);
LABEL_30:
        v22 = sub_B11F60(v17 + 80);
        v65 = sub_B12000(v17 + 72);
        if ( v22 )
          sub_AF47B0((__int64)&v66, *(unsigned __int64 **)(v22 + 16), *(unsigned __int64 **)(v22 + 24));
        else
          v67 = 0;
        v68 = v21;
        if ( v72[0] )
          sub_B91220((__int64)v72, v72[0]);
        if ( (v78 & 1) != 0 )
        {
          v8 = (__int64 *)&v79;
          v9 = 3;
LABEL_9:
          v69.m128i_i64[0] = 0;
          v70 = 0;
          v71 = 0;
          memset(v72, 0, 24);
          v72[3] = 1;
          v73 = 0;
          v62 = 0;
          if ( v67 )
            v62 = WORD4(v66) | ((_DWORD)v66 << 16);
          a2 = (__int64)&v62;
          v64 = v68;
          v63 = v65;
          v10 = sub_F11290(&v63, &v62, &v64);
          v12 = v65;
          v60 = 1;
          v57 = 0;
          v61 = v9 & v10;
          v56 = v9;
          v58 = v17;
          v13 = v65;
          while ( 1 )
          {
            v14 = (__int64)&v8[5 * v61];
            if ( *(_QWORD *)v14 == v13
              && v67 == *(_BYTE *)(v14 + 24)
              && (!v67 || v66 == *(_OWORD *)(v14 + 8))
              && v68 == *(_QWORD *)(v14 + 32) )
            {
              if ( *(_BYTE *)(v58 + 64) != 2 || (v49 = sub_B13870(v58), v50 = sub_AE9410(v49), v51 == v50) )
              {
                v47 = (unsigned int)v75;
                v48 = (unsigned int)v75 + 1LL;
                if ( v48 > HIDWORD(v75) )
                {
                  a2 = (__int64)v76;
                  sub_C8D5F0((__int64)&v74, v76, v48, 8u, v12, v11);
                  v47 = (unsigned int)v75;
                }
                *(_QWORD *)&v74[8 * v47] = v58;
                LODWORD(v75) = v75 + 1;
              }
              goto LABEL_22;
            }
            if ( sub_F34140((__int64)&v8[5 * v61], (__int64)&v69) )
              break;
            a2 = (__int64)v72;
            v53 = sub_F34140(v14, (__int64)v72);
            if ( !v57 )
            {
              if ( !v53 )
                v14 = 0;
              v57 = (__m128i *)v14;
            }
            v61 = v56 & (v60 + v61);
            ++v60;
          }
          v15 = v57;
          if ( !v57 )
            v15 = (__m128i *)&v8[5 * v61];
          ++v77;
          v72[0] = (__int64)v15;
          v16 = ((unsigned int)v78 >> 1) + 1;
          if ( (v78 & 1) == 0 )
          {
            a2 = v80;
            goto LABEL_37;
          }
          a2 = 4;
          if ( 4 * v16 >= 0xC )
            goto LABEL_38;
          goto LABEL_18;
        }
        a2 = v80;
        v8 = v79;
        v9 = v80 - 1;
        if ( v80 )
          goto LABEL_9;
        ++v77;
        v15 = 0;
        v72[0] = 0;
        v16 = ((unsigned int)v78 >> 1) + 1;
LABEL_37:
        if ( 3 * (int)a2 <= 4 * v16 )
        {
LABEL_38:
          LODWORD(a2) = 2 * a2;
LABEL_39:
          sub_F3E3C0((__int64)&v77, a2);
          a2 = (__int64)&v65;
          sub_F38D60((__int64)&v77, (__int64)&v65, v72);
          v15 = (__m128i *)v72[0];
          v16 = ((unsigned int)v78 >> 1) + 1;
          goto LABEL_19;
        }
LABEL_18:
        if ( (unsigned int)a2 - (v16 + HIDWORD(v78)) <= (unsigned int)a2 >> 3 )
          goto LABEL_39;
LABEL_19:
        LODWORD(v78) = v78 & 1 | (2 * v16);
        if ( v15->m128i_i64[0] || v15[1].m128i_i8[8] || v15[2].m128i_i64[0] )
          --HIDWORD(v78);
        *v15 = _mm_loadu_si128((const __m128i *)&v65);
        v15[1] = _mm_loadu_si128((const __m128i *)((char *)&v66 + 8));
        v15[2].m128i_i64[0] = v68;
LABEL_22:
        v7 = (_QWORD *)(*v7 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v7 == (_QWORD *)v59 )
          goto LABEL_77;
      }
    }
LABEL_99:
    v41 = 0;
    goto LABEL_70;
  }
  do
  {
    *v4 = 0;
    v4 += 5;
    *((_BYTE *)v4 - 16) = 0;
    *(v4 - 1) = 0;
  }
  while ( v4 != (__int64 *)v81 );
  v23 = (_QWORD *)(a1 + 48);
  v24 = (_QWORD *)(*(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (_QWORD *)(a1 + 48) == v24 )
    goto LABEL_99;
  do
  {
    while ( 1 )
    {
      if ( !v24 )
        BUG();
      if ( *((_BYTE *)v24 - 24) == 85 )
      {
        v25 = *(v24 - 7);
        if ( v25 )
        {
          if ( !*(_BYTE *)v25 && *(_QWORD *)(v25 + 24) == v24[7] && (*(_BYTE *)(v25 + 33) & 0x20) != 0 )
          {
            v26 = *(_DWORD *)(v25 + 36);
            if ( v26 == 71 || v26 == 68 )
              break;
          }
        }
      }
      sub_F38400((__int64)&v77);
LABEL_44:
      v24 = (_QWORD *)(*v24 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v23 == v24 )
        goto LABEL_67;
    }
    v27 = sub_B10CD0((__int64)(v24 + 3));
    v28 = *(_BYTE *)(v27 - 16);
    if ( (v28 & 2) == 0 )
    {
      if ( ((*(_WORD *)(v27 - 16) >> 6) & 0xF) != 2 )
        goto LABEL_55;
      v52 = v27 - 16 - 8LL * ((v28 >> 2) & 0xF);
LABEL_103:
      v29 = *(_QWORD *)(v52 + 8);
      goto LABEL_56;
    }
    if ( *(_DWORD *)(v27 - 24) == 2 )
    {
      v52 = *(_QWORD *)(v27 - 32);
      goto LABEL_103;
    }
LABEL_55:
    v29 = 0;
LABEL_56:
    v30 = v24 - 3;
    v31 = *((_DWORD *)v24 - 5) & 0x7FFFFFF;
    v32 = *(_QWORD *)(v24[4 * (2 - v31) - 3] + 24LL);
    v69.m128i_i64[0] = *(_QWORD *)(v24[4 * (1 - v31) - 3] + 24LL);
    if ( v32 )
      sub_AF47B0((__int64)&v69.m128i_i64[1], *(unsigned __int64 **)(v32 + 16), *(unsigned __int64 **)(v32 + 24));
    else
      v70 = 0;
    a2 = (__int64)&v77;
    v71 = v29;
    sub_F3E7D0((__int64)v72, (__int64)&v77, &v69);
    if ( (_BYTE)v73 )
      goto LABEL_44;
    v35 = *(v24 - 7);
    if ( !v35 || *(_BYTE *)v35 || *(_QWORD *)(v35 + 24) != v24[7] )
      BUG();
    if ( *(_DWORD *)(v35 + 36) == 68 )
    {
      v36 = sub_AE9410(*(_QWORD *)(v30[4 * (3LL - (*((_DWORD *)v24 - 5) & 0x7FFFFFF))] + 24LL));
      if ( v36 != v37 )
        goto LABEL_44;
    }
    v38 = (unsigned int)v75;
    v39 = (unsigned int)v75 + 1LL;
    if ( v39 > HIDWORD(v75) )
    {
      a2 = (__int64)v76;
      sub_C8D5F0((__int64)&v74, v76, v39, 8u, v33, v34);
      v38 = (unsigned int)v75;
    }
    *(_QWORD *)&v74[8 * v38] = v30;
    LODWORD(v75) = v75 + 1;
    v24 = (_QWORD *)(*v24 & 0xFFFFFFFFFFFFFFF8LL);
  }
  while ( v23 != v24 );
LABEL_67:
  v40 = (_QWORD **)v74;
  v2 = (_QWORD **)&v74[8 * (unsigned int)v75];
  v41 = v75;
  if ( v74 != (_BYTE *)v2 )
  {
    do
    {
      v42 = *v40++;
      sub_B43D60(v42);
    }
    while ( v2 != v40 );
LABEL_69:
    v41 = v75;
  }
LABEL_70:
  LOBYTE(v2) = v41 != 0;
  if ( (v78 & 1) == 0 )
  {
    a2 = 40LL * v80;
    sub_C7D6A0((__int64)v79, a2, 8);
  }
  if ( v74 != v76 )
    _libc_free(v74, a2);
  return (unsigned int)v2;
}
