// Function: sub_293D7E0
// Address: 0x293d7e0
//
__int64 __fastcall sub_293D7E0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r14
  unsigned __int64 v3; // r13
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  int v8; // eax
  unsigned __int64 v9; // rsi
  _QWORD *v10; // rax
  __int64 v11; // r12
  _BYTE *v12; // rdx
  _BYTE *v13; // rax
  _BYTE *i; // rsi
  __int64 v15; // rbx
  __int64 v16; // r13
  __m128i *v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rbx
  __int32 v25; // r12d
  _BYTE *v26; // rdx
  _QWORD *v27; // rax
  _QWORD *j; // rdx
  __int64 v29; // r9
  unsigned int v30; // ebx
  __int16 *v31; // rdx
  __int16 *v32; // rax
  __int16 *k; // rdx
  unsigned __int64 v34; // r12
  __int64 v35; // rax
  _BYTE *v36; // rbx
  unsigned __int64 v37; // r12
  unsigned __int64 v38; // rdi
  __m128i v39; // rax
  char v40; // al
  _QWORD *v41; // rcx
  __int64 *v42; // r12
  unsigned int v43; // r13d
  __int64 v44; // r14
  __int64 (__fastcall *v45)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v46; // rax
  unsigned __int8 *v47; // r10
  __int64 v48; // r11
  _BYTE **v49; // rcx
  __int64 v50; // r14
  _BYTE *v51; // rdi
  __int64 v52; // r11
  unsigned int v53; // r13d
  char *v54; // r13
  char *v55; // r12
  __int64 v56; // rdx
  unsigned int v57; // esi
  __int64 *v58; // rsi
  __int64 *v59; // rax
  __int64 v60; // rcx
  int v61; // edx
  char v62; // dl
  __m128i v63; // xmm3
  _QWORD *v64; // [rsp+8h] [rbp-898h]
  unsigned int v65; // [rsp+3Ch] [rbp-864h]
  __int64 v66; // [rsp+50h] [rbp-850h]
  __int64 v67; // [rsp+58h] [rbp-848h]
  __int64 v68; // [rsp+60h] [rbp-840h]
  unsigned __int64 v69; // [rsp+68h] [rbp-838h]
  __int64 v70; // [rsp+70h] [rbp-830h]
  __int64 v71; // [rsp+80h] [rbp-820h]
  unsigned __int8 v72; // [rsp+8Bh] [rbp-815h]
  unsigned int v73; // [rsp+8Ch] [rbp-814h]
  __int64 v74; // [rsp+98h] [rbp-808h]
  char v75[8]; // [rsp+A0h] [rbp-800h] BYREF
  int v76; // [rsp+A8h] [rbp-7F8h]
  unsigned __int32 v77; // [rsp+ACh] [rbp-7F4h]
  unsigned __int8 v78; // [rsp+C0h] [rbp-7E0h]
  __m128i v79; // [rsp+D0h] [rbp-7D0h] BYREF
  __m128i v80; // [rsp+E0h] [rbp-7C0h] BYREF
  __int64 v81; // [rsp+F0h] [rbp-7B0h]
  _QWORD v82[4]; // [rsp+100h] [rbp-7A0h] BYREF
  __int16 v83; // [rsp+120h] [rbp-780h]
  __m128i v84; // [rsp+130h] [rbp-770h] BYREF
  __m128i v85; // [rsp+140h] [rbp-760h]
  __int64 v86; // [rsp+150h] [rbp-750h]
  unsigned __int64 v87; // [rsp+160h] [rbp-740h] BYREF
  unsigned int v88; // [rsp+168h] [rbp-738h]
  unsigned __int64 v89; // [rsp+170h] [rbp-730h]
  unsigned int v90; // [rsp+178h] [rbp-728h]
  __int16 v91; // [rsp+180h] [rbp-720h]
  _QWORD *v92; // [rsp+190h] [rbp-710h] BYREF
  __int64 v93; // [rsp+198h] [rbp-708h]
  _QWORD v94[8]; // [rsp+1A0h] [rbp-700h] BYREF
  __m128i v95; // [rsp+1E0h] [rbp-6C0h] BYREF
  _BYTE v96[64]; // [rsp+1F0h] [rbp-6B0h] BYREF
  char *v97; // [rsp+230h] [rbp-670h] BYREF
  int v98; // [rsp+238h] [rbp-668h]
  char v99; // [rsp+240h] [rbp-660h] BYREF
  __int64 v100; // [rsp+268h] [rbp-638h]
  __int64 v101; // [rsp+270h] [rbp-630h]
  __int64 v102; // [rsp+280h] [rbp-620h]
  __int64 v103; // [rsp+288h] [rbp-618h]
  void *v104; // [rsp+2B0h] [rbp-5F0h]
  __m128i v105; // [rsp+2C0h] [rbp-5E0h] BYREF
  __int16 v106; // [rsp+2D0h] [rbp-5D0h] BYREF
  __int64 v107; // [rsp+2D8h] [rbp-5C8h]
  __m128i v108; // [rsp+2E0h] [rbp-5C0h] BYREF
  __m128i v109; // [rsp+2F0h] [rbp-5B0h] BYREF
  __int8 v110; // [rsp+300h] [rbp-5A0h]
  __int64 v111; // [rsp+308h] [rbp-598h]
  char *v112; // [rsp+310h] [rbp-590h] BYREF
  char v113; // [rsp+320h] [rbp-580h] BYREF
  _BYTE *v114; // [rsp+360h] [rbp-540h] BYREF
  __int64 v115; // [rsp+368h] [rbp-538h]
  _BYTE v116[1328]; // [rsp+370h] [rbp-530h] BYREF

  v2 = (_QWORD *)a1;
  v3 = a2;
  if ( *(_DWORD *)(a1 + 1152) && !sub_293A020(a1, (unsigned __int8 *)a2) )
    return 0;
  sub_2939E80((__int64)v75, a1, *(_QWORD *)(a2 + 8));
  v72 = v78;
  if ( !v78 )
    return 0;
  sub_23D0AB0((__int64)&v97, a2, 0, 0, 0);
  v8 = *(_DWORD *)(a2 + 4);
  v93 = 0x800000000LL;
  v9 = v8 & 0x7FFFFFF;
  v73 = v8 & 0x7FFFFFF;
  v10 = v94;
  v69 = v9;
  v92 = v94;
  if ( !v9 )
  {
    v115 = 0x800000000LL;
    v114 = v116;
    goto LABEL_25;
  }
  v11 = v9;
  v12 = &v94[v9];
  if ( v9 > 8 )
  {
    sub_C8D5F0((__int64)&v92, v94, v9, 8u, v6, v7);
    v10 = &v92[(unsigned int)v93];
    v12 = &v92[v11];
    if ( &v92[v11] == v10 )
    {
      v115 = 0x800000000LL;
      LODWORD(v93) = v73;
      v114 = v116;
LABEL_62:
      sub_293A5B0((__int64)&v114, v9, (__int64)v12, v5, v6, v7);
      v13 = v114;
      v12 = &v114[160 * (unsigned int)v115];
      goto LABEL_11;
    }
  }
  do
  {
    if ( v10 )
      *v10 = 0;
    ++v10;
  }
  while ( v10 != (_QWORD *)v12 );
  v115 = 0x800000000LL;
  LODWORD(v93) = v73;
  v13 = v116;
  v12 = v116;
  v114 = v116;
  if ( v9 > 8 )
    goto LABEL_62;
LABEL_11:
  for ( i = &v13[160 * v9]; i != v12; v12 += 160 )
  {
    if ( v12 )
    {
      memset(v12, 0, 0xA0u);
      *((_DWORD *)v12 + 23) = 8;
      *((_QWORD *)v12 + 10) = v12 + 96;
    }
  }
  LODWORD(v115) = v73;
  if ( v73 )
  {
    v15 = v3;
    v16 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v22 = *(_QWORD *)(v15 + 32 * (v16 - (*(_DWORD *)(v15 + 4) & 0x7FFFFFF)));
        v23 = *(_QWORD *)(v22 + 8);
        if ( *(_BYTE *)(v23 + 8) == 17 )
          break;
        v92[v16++] = v22;
        if ( v73 <= (unsigned int)v16 )
          goto LABEL_24;
      }
      sub_2939E80((__int64)&v95, (__int64)v2, v23);
      if ( !v96[16] || v95.m128i_i32[2] != v76 )
        break;
      sub_293CE40(&v105, v2, v15, *(_QWORD *)(v15 + 32 * (v16 - (*(_DWORD *)(v15 + 4) & 0x7FFFFFF))), &v95);
      v17 = (__m128i *)&v114[160 * v16];
      v17->m128i_i64[0] = v105.m128i_i64[0];
      v17 += 5;
      v17[-5].m128i_i64[1] = v105.m128i_i64[1];
      v17[-4].m128i_i16[0] = v106;
      v17[-4].m128i_i64[1] = v107;
      v17[-3] = _mm_loadu_si128(&v108);
      v17[-2] = _mm_loadu_si128(&v109);
      v17[-1].m128i_i8[0] = v110;
      v18 = v111;
      v17[-1].m128i_i64[1] = v111;
      sub_293A290((__int64)v17, &v112, v18, v19, v20, v21);
      if ( v112 != &v113 )
        _libc_free((unsigned __int64)v112);
      if ( v73 <= (unsigned int)++v16 )
      {
LABEL_24:
        v3 = v15;
        goto LABEL_25;
      }
    }
    v72 = 0;
    goto LABEL_49;
  }
LABEL_25:
  v24 = v77;
  v95.m128i_i64[0] = (__int64)v96;
  v25 = v77;
  v95.m128i_i64[1] = 0x800000000LL;
  if ( v77 )
  {
    v26 = v96;
    v27 = v96;
    if ( v77 > 8uLL )
    {
      sub_C8D5F0((__int64)&v95, v96, v77, 8u, v6, v7);
      v26 = (_BYTE *)v95.m128i_i64[0];
      v27 = (_QWORD *)(v95.m128i_i64[0] + 8LL * v95.m128i_u32[2]);
    }
    for ( j = &v26[8 * v24]; j != v27; ++v27 )
    {
      if ( v27 )
        *v27 = 0;
    }
    v29 = v77;
    v95.m128i_i32[2] = v25;
    if ( v77 )
    {
      v64 = v2;
      v30 = 0;
      v71 = v3;
      while ( 1 )
      {
        v105.m128i_i64[0] = (__int64)&v106;
        v105.m128i_i64[1] = 0x800000000LL;
        if ( v69 )
        {
          v31 = &v106;
          v32 = &v106;
          if ( v69 > 8 )
          {
            sub_C8D5F0((__int64)&v105, &v106, v69, 8u, v6, v29);
            v31 = (__int16 *)v105.m128i_i64[0];
            v32 = (__int16 *)(v105.m128i_i64[0] + 8LL * v105.m128i_u32[2]);
          }
          for ( k = &v31[4 * v69]; k != v32; v32 += 4 )
          {
            if ( v32 )
              *(_QWORD *)v32 = 0;
          }
          v105.m128i_i32[2] = v73;
          if ( v73 )
          {
            v34 = 0;
            do
            {
              v35 = v92[v34 / 8];
              if ( !v35 )
                v35 = sub_293BC00((__int64)&v114[20 * v34], v30);
              *(_QWORD *)(v105.m128i_i64[0] + v34) = v35;
              v34 += 8LL;
            }
            while ( 8LL * (v73 - 1) + 8 != v34 );
          }
        }
        LODWORD(v82[0]) = v30;
        v83 = 265;
        v39.m128i_i64[0] = (__int64)sub_BD5D20(v71);
        v79 = v39;
        v80.m128i_i64[0] = (__int64)".i";
        v40 = v83;
        LOWORD(v81) = 773;
        if ( (_BYTE)v83 )
        {
          if ( (_BYTE)v83 == 1 )
          {
            v63 = _mm_loadu_si128(&v80);
            v84 = _mm_loadu_si128(&v79);
            v86 = v81;
            v85 = v63;
          }
          else
          {
            if ( HIBYTE(v83) == 1 )
            {
              v41 = (_QWORD *)v82[0];
              v66 = v82[1];
            }
            else
            {
              v41 = v82;
              v40 = 2;
            }
            v85.m128i_i64[0] = (__int64)v41;
            v84.m128i_i64[0] = (__int64)&v79;
            v85.m128i_i64[1] = v66;
            LOBYTE(v86) = 2;
            BYTE1(v86) = v40;
          }
        }
        else
        {
          LOWORD(v86) = 256;
        }
        v42 = (__int64 *)(v105.m128i_i64[0] + 8);
        v43 = v105.m128i_u32[2];
        v44 = *(_QWORD *)(v71 + 72);
        v68 = v105.m128i_u32[2] - 1LL;
        v70 = *(_QWORD *)v105.m128i_i64[0];
        v45 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v102 + 64LL);
        v67 = v44;
        if ( v45 == sub_920540 )
        {
          if ( sub_BCEA30(v44)
            || *(_BYTE *)v70 > 0x15u
            || (v46 = sub_293A090((_BYTE **)v42, (__int64)&v42[v68]), v49 != v46) )
          {
LABEL_84:
            v91 = 257;
            v50 = (__int64)sub_BD2C40(88, v43);
            if ( v50 )
            {
              v52 = *(_QWORD *)(v70 + 8);
              v53 = v65 & 0xE0000000 | v43 & 0x7FFFFFF;
              v65 = v53;
              if ( (unsigned int)*(unsigned __int8 *)(v52 + 8) - 17 > 1 )
              {
                v58 = &v42[v68];
                if ( v42 != v58 )
                {
                  v59 = v42;
                  v60 = *(_QWORD *)(*v42 + 8);
                  v61 = *(unsigned __int8 *)(v60 + 8);
                  if ( v61 == 17 )
                  {
LABEL_96:
                    v62 = 0;
                  }
                  else
                  {
                    while ( v61 != 18 )
                    {
                      if ( v58 == ++v59 )
                        goto LABEL_86;
                      v60 = *(_QWORD *)(*v59 + 8);
                      v61 = *(unsigned __int8 *)(v60 + 8);
                      if ( v61 == 17 )
                        goto LABEL_96;
                    }
                    v62 = 1;
                  }
                  BYTE4(v74) = v62;
                  LODWORD(v74) = *(_DWORD *)(v60 + 32);
                  v52 = sub_BCE1B0((__int64 *)v52, v74);
                }
              }
LABEL_86:
              sub_B44260(v50, v52, 34, v53, 0, 0);
              *(_QWORD *)(v50 + 72) = v67;
              *(_QWORD *)(v50 + 80) = sub_B4DC50(v67, (__int64)v42, v68);
              sub_B4D9A0(v50, v70, v42, v68, (__int64)&v87);
            }
            sub_B4DDE0(v50, 0);
            (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v103 + 16LL))(
              v103,
              v50,
              &v84,
              v100,
              v101);
            v54 = v97;
            v55 = &v97[16 * v98];
            if ( v97 != v55 )
            {
              do
              {
                v56 = *((_QWORD *)v54 + 1);
                v57 = *(_DWORD *)v54;
                v54 += 16;
                sub_B99FD0(v50, v57, v56);
              }
              while ( v55 != v54 );
            }
            goto LABEL_74;
          }
          LOBYTE(v91) = 0;
          v50 = sub_AD9FD0(v44, v47, v42, v48, 0, (__int64)&v87, 0);
          if ( (_BYTE)v91 )
          {
            LOBYTE(v91) = 0;
            if ( v90 > 0x40 && v89 )
              j_j___libc_free_0_0(v89);
            if ( v88 > 0x40 && v87 )
              j_j___libc_free_0_0(v87);
          }
        }
        else
        {
          v50 = v45(v102, v44, (_BYTE *)v70, (_BYTE **)(v105.m128i_i64[0] + 8), v68, 0);
        }
        if ( !v50 )
          goto LABEL_84;
LABEL_74:
        *(_QWORD *)(v95.m128i_i64[0] + 8LL * v30) = v50;
        if ( sub_B4DE30(v71) )
        {
          v51 = *(_BYTE **)(v95.m128i_i64[0] + 8LL * v30);
          if ( *v51 == 63 )
            sub_B4DE00((__int64)v51, 1);
        }
        if ( (__int16 *)v105.m128i_i64[0] != &v106 )
          _libc_free(v105.m128i_u64[0]);
        if ( v77 <= ++v30 )
        {
          v2 = v64;
          v3 = v71;
          break;
        }
      }
    }
  }
  sub_293CAB0((__int64)v2, v3, (__int64)&v95, (__int64)v75);
  if ( (_BYTE *)v95.m128i_i64[0] != v96 )
    _libc_free(v95.m128i_u64[0]);
LABEL_49:
  v36 = v114;
  v37 = (unsigned __int64)&v114[160 * (unsigned int)v115];
  if ( v114 != (_BYTE *)v37 )
  {
    do
    {
      v37 -= 160LL;
      v38 = *(_QWORD *)(v37 + 80);
      if ( v38 != v37 + 96 )
        _libc_free(v38);
    }
    while ( v36 != (_BYTE *)v37 );
    v37 = (unsigned __int64)v114;
  }
  if ( (_BYTE *)v37 != v116 )
    _libc_free(v37);
  if ( v92 != v94 )
    _libc_free((unsigned __int64)v92);
  nullsub_61();
  v104 = &unk_49DA100;
  nullsub_63();
  if ( v97 != &v99 )
    _libc_free((unsigned __int64)v97);
  return v72;
}
