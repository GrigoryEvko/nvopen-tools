// Function: sub_3281800
// Address: 0x3281800
//
__int64 __fastcall sub_3281800(__int64 a1, __int64 a2, unsigned int a3, __m128i *a4, unsigned int a5)
{
  __int64 v6; // r8
  unsigned __int16 v8; // ax
  __int64 v13; // rax
  __int64 v14; // rdx
  char v15; // al
  char v16; // dl
  unsigned int v17; // eax
  unsigned __int16 v18; // r9
  __int64 v19; // rsi
  char v20; // cl
  unsigned __int16 v21; // dx
  char v22; // al
  __int64 v23; // rax
  int v24; // edx
  unsigned int *v25; // rax
  unsigned int *v26; // rax
  __int16 v27; // ax
  __int64 v28; // rax
  int v29; // edx
  bool v30; // al
  __int64 v31; // rax
  unsigned __int64 v32; // rcx
  __int64 v33; // rdx
  char v34; // si
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rdi
  unsigned __int64 v38; // rdx
  char v39; // al
  char v40; // al
  char v41; // cl
  unsigned __int64 v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rax
  unsigned __int16 v45; // ax
  __int64 v46; // rdx
  __int128 v47; // rax
  __int64 v48; // rax
  unsigned __int8 v49; // r8
  unsigned __int64 v50; // r14
  unsigned __int16 v51; // ax
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // rsi
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rdi
  __int64 (__fastcall *v61)(__int64, __int64, __int64, __int64, __int64); // rax
  __int64 v62; // rax
  int v63; // edx
  bool v64; // al
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rdx
  __int64 v69; // rax
  char v70; // [rsp-F1h] [rbp-F1h]
  unsigned __int8 v71; // [rsp-F1h] [rbp-F1h]
  __int64 v72; // [rsp-F0h] [rbp-F0h]
  __int64 v73; // [rsp-E8h] [rbp-E8h]
  __int64 v74; // [rsp-E8h] [rbp-E8h]
  unsigned int v75; // [rsp-E8h] [rbp-E8h]
  __int64 v76; // [rsp-E0h] [rbp-E0h]
  __int64 v77; // [rsp-E0h] [rbp-E0h]
  unsigned int v78; // [rsp-E0h] [rbp-E0h]
  unsigned __int16 v79; // [rsp-D8h] [rbp-D8h]
  unsigned __int16 v80; // [rsp-D8h] [rbp-D8h]
  unsigned __int64 v81; // [rsp-D8h] [rbp-D8h]
  __int64 (__fastcall *v82)(__int64, _QWORD, __int64, _QWORD, __int64, _QWORD, _QWORD, _QWORD, _QWORD); // [rsp-D8h] [rbp-D8h]
  unsigned __int8 v83; // [rsp-D8h] [rbp-D8h]
  unsigned __int64 v84; // [rsp-D8h] [rbp-D8h]
  unsigned __int8 v85; // [rsp-D0h] [rbp-D0h]
  unsigned __int8 v86; // [rsp-D0h] [rbp-D0h]
  __int64 v87; // [rsp-C8h] [rbp-C8h]
  char v88; // [rsp-C0h] [rbp-C0h]
  unsigned __int16 v89; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v90; // [rsp-B0h] [rbp-B0h]
  __int64 v91; // [rsp-A8h] [rbp-A8h] BYREF
  char v92; // [rsp-A0h] [rbp-A0h]
  unsigned __int16 v93; // [rsp-98h] [rbp-98h] BYREF
  __int64 v94; // [rsp-90h] [rbp-90h]
  __int64 v95; // [rsp-88h] [rbp-88h] BYREF
  __int64 v96; // [rsp-80h] [rbp-80h]
  __int64 v97; // [rsp-78h] [rbp-78h]
  __int64 v98; // [rsp-70h] [rbp-70h]
  __int64 v99; // [rsp-68h] [rbp-68h]
  __int64 v100; // [rsp-60h] [rbp-60h]
  __int64 v101; // [rsp-58h] [rbp-58h] BYREF
  __int64 v102; // [rsp-50h] [rbp-50h]
  _OWORD v103[4]; // [rsp-48h] [rbp-48h] BYREF

  if ( !a2 )
    return 0;
  if ( (a5 & 7) != 0 )
    goto LABEL_3;
  v8 = a4->m128i_i16[0];
  if ( a4->m128i_i16[0] )
  {
    if ( (unsigned __int16)(v8 - 176) <= 0x34u )
      goto LABEL_3;
    if ( v8 == 1 || (unsigned __int16)(v8 - 504) <= 7u )
      goto LABEL_110;
    v13 = 16LL * (v8 - 1);
    v14 = *(_QWORD *)&byte_444C4A0[v13];
    v15 = byte_444C4A0[v13 + 8];
  }
  else
  {
    if ( sub_3007100((__int64)a4) )
      goto LABEL_3;
    v87 = sub_3007260((__int64)a4);
    v88 = v16;
    v14 = v87;
    v15 = v88;
  }
  BYTE8(v103[0]) = v15;
  *(_QWORD *)&v103[0] = v14;
  v17 = sub_CA1930(v103);
  if ( v17 <= 7 )
    goto LABEL_3;
  if ( (v17 & (v17 - 1)) != 0 )
    goto LABEL_3;
  v6 = *(_QWORD *)(a2 + 112);
  if ( (*(_BYTE *)(v6 + 37) & 0xF) != 0 || (*(_BYTE *)(a2 + 32) & 8) != 0 )
    goto LABEL_3;
  v18 = *(_WORD *)(a2 + 96);
  v19 = *(_QWORD *)(a2 + 104);
  v89 = v18;
  v90 = v19;
  if ( v18 )
  {
    v20 = (unsigned __int16)(v18 - 176) <= 0x34u;
  }
  else
  {
    v74 = v6;
    v30 = sub_3007100((__int64)&v89);
    v6 = v74;
    v18 = 0;
    v20 = v30;
  }
  v21 = a4->m128i_i16[0];
  if ( a4->m128i_i16[0] )
  {
    v22 = (unsigned __int16)(v21 - 176) <= 0x34u;
  }
  else
  {
    v70 = v20;
    v73 = v6;
    v79 = v18;
    v22 = sub_3007100((__int64)a4);
    v20 = v70;
    v21 = 0;
    v6 = v73;
    v18 = v79;
  }
  if ( v22 != v20 )
    goto LABEL_3;
  v23 = a4->m128i_i64[1];
  if ( v18 == v21 )
  {
    if ( v18 || v19 == v23 )
      goto LABEL_23;
    v102 = a4->m128i_i64[1];
    LOWORD(v101) = 0;
  }
  else
  {
    LOWORD(v101) = v21;
    v102 = v23;
    if ( v21 )
    {
      if ( v21 == 1 || (unsigned __int16)(v21 - 504) <= 7u )
        goto LABEL_110;
      v32 = *(_QWORD *)&byte_444C4A0[16 * v21 - 16];
      v34 = byte_444C4A0[16 * v21 - 8];
      goto LABEL_44;
    }
  }
  v76 = v6;
  v80 = v18;
  v31 = sub_3007260((__int64)&v101);
  v6 = v76;
  v18 = v80;
  v99 = v31;
  v32 = v31;
  v100 = v33;
  v34 = v33;
LABEL_44:
  if ( v18 )
  {
    if ( v18 == 1 || (unsigned __int16)(v18 - 504) <= 7u )
      goto LABEL_110;
    v38 = *(_QWORD *)&byte_444C4A0[16 * v18 - 16];
    v39 = byte_444C4A0[16 * v18 - 8];
  }
  else
  {
    v77 = v6;
    v81 = v32;
    v35 = sub_3007260((__int64)&v89);
    v6 = v77;
    v37 = v36;
    v97 = v35;
    v38 = v35;
    v32 = v81;
    v98 = v37;
    v39 = v37;
  }
  if ( (!v39 || v34) && v38 < v32 )
    goto LABEL_3;
LABEL_23:
  if ( a5 )
  {
    v40 = sub_2EAC4F0(v6);
    v41 = -1;
    v42 = -(__int64)((a5 >> 3) | (unsigned __int64)(1LL << v40)) & ((a5 >> 3) | (unsigned __int64)(1LL << v40));
    if ( v42 )
    {
      _BitScanReverse64(&v42, v42);
      v41 = 63 - (v42 ^ 0x3F);
    }
    v43 = *(_QWORD *)(a2 + 112);
    v71 = v41;
    v72 = *(_QWORD *)(a1 + 8);
    v75 = *(unsigned __int16 *)(v43 + 32);
    v82 = *(__int64 (__fastcall **)(__int64, _QWORD, __int64, _QWORD, __int64, _QWORD, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v72 + 824LL);
    v78 = sub_2EAC1E0(v43);
    v44 = sub_2E79000(*(__int64 **)(*(_QWORD *)a1 + 40LL));
    LODWORD(v6) = v82(v72, *(_QWORD *)(*(_QWORD *)a1 + 64LL), v44, a4->m128i_u32[0], a4->m128i_i64[1], v78, v71, v75, 0);
    if ( !(_BYTE)v6 )
      return (unsigned int)v6;
  }
  v24 = *(_DWORD *)(a2 + 24);
  v25 = *(unsigned int **)(a2 + 40);
  if ( v24 <= 365 )
  {
    if ( v24 <= 363 )
    {
      if ( v24 != 339 && (v24 & 0xFFFFFFBF) != 0x12B )
        goto LABEL_28;
      goto LABEL_54;
    }
LABEL_52:
    v26 = v25 + 30;
    goto LABEL_29;
  }
  if ( v24 > 467 )
  {
    if ( v24 == 497 )
      goto LABEL_52;
LABEL_28:
    v26 = v25 + 10;
    goto LABEL_29;
  }
  if ( v24 <= 464 )
    goto LABEL_28;
LABEL_54:
  v26 = v25 + 20;
LABEL_29:
  v27 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)v26 + 48LL) + 16LL * v26[2]);
  LOBYTE(v6) = v27 == 0 || v27 == 264;
  if ( (_BYTE)v6 )
    goto LABEL_3;
  if ( v24 != 298 )
  {
    v45 = *(_WORD *)(a2 + 96);
    v46 = *(_QWORD *)(a2 + 104);
    v93 = v45;
    v94 = v46;
    if ( v45 )
    {
      if ( v45 == 1 || (unsigned __int16)(v45 - 504) <= 7u )
        goto LABEL_110;
      *((_QWORD *)&v47 + 1) = 16LL * (v45 - 1);
      *(_QWORD *)&v47 = *(_QWORD *)&byte_444C4A0[*((_QWORD *)&v47 + 1)];
      BYTE8(v47) = byte_444C4A0[*((_QWORD *)&v47 + 1) + 8];
    }
    else
    {
      *(_QWORD *)&v47 = sub_3007260((__int64)&v93);
      LOBYTE(v6) = 0;
      v103[0] = v47;
    }
    v85 = v6;
    v91 = v47;
    v92 = BYTE8(v47);
    v48 = sub_CA1930(&v91);
    v49 = v85;
    v50 = v48;
    v51 = a4->m128i_i16[0];
    if ( !a4->m128i_i16[0] )
    {
      v52 = sub_3007260((__int64)a4);
      v49 = v85;
      v101 = v52;
      v102 = v53;
LABEL_71:
      v86 = v49;
      v95 = v52;
      LOBYTE(v96) = v53;
      if ( v50 < sub_CA1930(&v95) + (unsigned __int64)a5 )
        goto LABEL_3;
      if ( *(_BYTE *)(a1 + 33) )
      {
        LODWORD(v6) = v86;
        v54 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 48LL)
                                  + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 48LL));
        if ( (_WORD)v54 )
        {
          v55 = *(_QWORD *)(a1 + 8);
          v56 = (unsigned __int16)v54;
          if ( *(_QWORD *)(v55 + 8 * v54 + 112) )
          {
            v57 = a4->m128i_u16[0];
            if ( (_WORD)v57 )
            {
              if ( !*(_BYTE *)(v57 + v55 + 274 * v56 + 443718) )
                LODWORD(v6) = *(unsigned __int8 *)(a1 + 33);
            }
          }
        }
        return (unsigned int)v6;
      }
      goto LABEL_98;
    }
    if ( v51 != 1 && (unsigned __int16)(v51 - 504) > 7u )
    {
      v53 = 16LL * (v51 - 1);
      v52 = *(_QWORD *)&byte_444C4A0[v53];
      LOBYTE(v53) = byte_444C4A0[v53 + 8];
      goto LABEL_71;
    }
LABEL_110:
    BUG();
  }
  v28 = *(_QWORD *)(a2 + 56);
  if ( !v28 )
    return (unsigned int)v6;
  v29 = 1;
  do
  {
    if ( !*(_DWORD *)(v28 + 8) )
    {
      if ( !v29 )
        return (unsigned int)v6;
      v28 = *(_QWORD *)(v28 + 32);
      if ( !v28 )
        goto LABEL_80;
      if ( !*(_DWORD *)(v28 + 8) )
        return (unsigned int)v6;
      v29 = 0;
    }
    v28 = *(_QWORD *)(v28 + 32);
  }
  while ( v28 );
  if ( v29 == 1 )
    goto LABEL_3;
LABEL_80:
  if ( *(_BYTE *)(a1 + 33)
    && ((v58 = a4->m128i_u16[0], v59 = **(unsigned __int16 **)(a2 + 48), !(_WORD)v59)
     || !(_WORD)v58
     || (((int)*(unsigned __int16 *)(*(_QWORD *)(a1 + 8) + 2 * (v58 + 274 * v59 + 71704) + 6) >> (4 * a3)) & 0xF) != 0)
    || *(_DWORD *)(a2 + 68) > 2u )
  {
LABEL_3:
    LODWORD(v6) = 0;
    return (unsigned int)v6;
  }
  if ( (*(_BYTE *)(a2 + 33) & 0xC) != 0 )
  {
    v65 = *(_QWORD *)(a2 + 104);
    LOWORD(v101) = *(_WORD *)(a2 + 96);
    v102 = v65;
    v66 = sub_2D5B750((unsigned __int16 *)&v101);
    v96 = v67;
    v95 = v66;
    v84 = sub_CA1930(&v95);
    *(_QWORD *)&v103[0] = sub_2D5B750((unsigned __int16 *)a4);
    *((_QWORD *)&v103[0] + 1) = v68;
    v69 = sub_CA1930(v103);
    LODWORD(v6) = 0;
    if ( v84 < v69 + (unsigned __int64)a5 )
      return (unsigned int)v6;
  }
  v60 = *(_QWORD *)(a1 + 8);
  v61 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v60 + 776LL);
  if ( v61 == sub_2FE41D0 )
  {
    v103[0] = _mm_loadu_si128(a4);
    if ( LOWORD(v103[0]) )
    {
      if ( (unsigned __int16)(LOWORD(v103[0]) - 17) <= 0xD3u )
      {
LABEL_89:
        v62 = *(_QWORD *)(a2 + 56);
        if ( !v62 )
          return (unsigned int)v6;
        v63 = 1;
        do
        {
          if ( !*(_DWORD *)(v62 + 8) )
          {
            if ( !v63 )
              return (unsigned int)v6;
            v62 = *(_QWORD *)(v62 + 32);
            if ( !v62 )
              goto LABEL_98;
            if ( !*(_DWORD *)(v62 + 8) )
              return (unsigned int)v6;
            v63 = 0;
          }
          v62 = *(_QWORD *)(v62 + 32);
        }
        while ( v62 );
        if ( v63 == 1 )
          return (unsigned int)v6;
      }
    }
    else
    {
      v83 = v6;
      v64 = sub_30070B0((__int64)v103);
      LODWORD(v6) = v83;
      if ( v64 )
        goto LABEL_89;
    }
LABEL_98:
    LODWORD(v6) = 1;
    return (unsigned int)v6;
  }
  return v61(v60, a2, a3, a4->m128i_u32[0], a4->m128i_i64[1]);
}
