// Function: sub_28AFC20
// Address: 0x28afc20
//
__int64 __fastcall sub_28AFC20(__int64 a1, __int64 a2, _WORD *a3)
{
  unsigned __int8 *v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  _BYTE *v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r9
  unsigned int v16; // r15d
  __int64 v17; // rax
  __int64 v19; // rax
  __int64 v20; // rcx
  int v21; // eax
  __int64 v22; // rsi
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // rdi
  _BYTE *v26; // r13
  unsigned __int8 *v27; // rax
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 *v30; // rax
  __int64 *v31; // rax
  __m128i **v32; // rax
  __m128i *v33; // r15
  _QWORD *v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdx
  __m128i *v37; // rsi
  __int64 v38; // rax
  __int64 v39; // r15
  __int64 v40; // r10
  __int64 v41; // rax
  __int64 v42; // rdx
  _BYTE *v43; // r8
  unsigned __int16 v44; // ax
  unsigned __int8 v45; // r9
  _QWORD *v46; // rax
  unsigned __int8 *v47; // rax
  unsigned int v48; // eax
  __int64 v49; // rdx
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // rcx
  unsigned __int8 *v53; // rax
  unsigned __int8 *v54; // r15
  int v55; // eax
  __int64 v56; // rax
  __int64 v57; // rax
  unsigned int v58; // eax
  __int64 v59; // rax
  __int64 *v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rdx
  unsigned __int8 *v66; // rax
  __int64 v67; // rcx
  _QWORD *v68; // rdx
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rcx
  int v74; // r8d
  _QWORD *v75; // [rsp+10h] [rbp-370h]
  __int64 v76; // [rsp+10h] [rbp-370h]
  unsigned __int8 v77; // [rsp+10h] [rbp-370h]
  __int64 (__fastcall *v78)(_QWORD *, __m128i *, __m128i *, __m128i *); // [rsp+18h] [rbp-368h]
  __int64 v79; // [rsp+18h] [rbp-368h]
  unsigned __int8 v80; // [rsp+18h] [rbp-368h]
  unsigned __int8 *v81; // [rsp+20h] [rbp-360h]
  __int64 v82; // [rsp+20h] [rbp-360h]
  __int64 v83; // [rsp+20h] [rbp-360h]
  _BYTE *v84; // [rsp+20h] [rbp-360h]
  __int64 v85; // [rsp+28h] [rbp-358h]
  __int64 v86; // [rsp+28h] [rbp-358h]
  char v87; // [rsp+30h] [rbp-350h]
  __m128i v88[3]; // [rsp+40h] [rbp-340h] BYREF
  __m128i v89; // [rsp+70h] [rbp-310h] BYREF
  __int64 (__fastcall *v90)(__m128i *, __m128i *, int); // [rsp+80h] [rbp-300h]
  __int64 (__fastcall *v91)(__int64); // [rsp+88h] [rbp-2F8h]
  __m128i v92; // [rsp+A0h] [rbp-2E0h] BYREF
  __int64 v93; // [rsp+B0h] [rbp-2D0h] BYREF
  __int64 v94; // [rsp+B8h] [rbp-2C8h]
  __int64 v95; // [rsp+C0h] [rbp-2C0h] BYREF
  __int64 v96; // [rsp+C8h] [rbp-2B8h]
  __int64 v97; // [rsp+D0h] [rbp-2B0h]
  __int64 v98; // [rsp+D8h] [rbp-2A8h]
  __int16 v99; // [rsp+E0h] [rbp-2A0h]
  void *v100; // [rsp+120h] [rbp-260h]
  _QWORD v101[2]; // [rsp+200h] [rbp-180h] BYREF
  char v102; // [rsp+210h] [rbp-170h]
  _BYTE *v103; // [rsp+218h] [rbp-168h]
  __int64 v104; // [rsp+220h] [rbp-160h]
  _BYTE v105[128]; // [rsp+228h] [rbp-158h] BYREF
  __int16 v106; // [rsp+2A8h] [rbp-D8h]
  void *v107; // [rsp+2B0h] [rbp-D0h]
  __int64 v108; // [rsp+2B8h] [rbp-C8h]
  __int64 v109; // [rsp+2C0h] [rbp-C0h]
  __int64 v110; // [rsp+2C8h] [rbp-B8h] BYREF
  unsigned int v111; // [rsp+2D0h] [rbp-B0h]
  char v112; // [rsp+348h] [rbp-38h] BYREF

  v6 = sub_BD3990(*(unsigned __int8 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), a2);
  if ( v6 == sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), a2) )
    goto LABEL_7;
  v7 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v11 = *(_BYTE **)(a2 + 32 * (2 - v7));
  LOBYTE(v12) = *v11;
  if ( *v11 > 0x1Cu )
  {
    v92 = (__m128i)(unsigned __int64)sub_B43CC0(*(_QWORD *)(a2 + 32 * (2 - v7)));
    v93 = 0;
    v94 = 0;
    v95 = 0;
    v96 = 0;
    v97 = 0;
    v98 = 0;
    v99 = 257;
    v12 = sub_1020E10((__int64)v11, &v92, v13, v14, 257, v15);
    v7 = v12;
    if ( v12 )
    {
      LOBYTE(v12) = *(_BYTE *)v12;
      v11 = (_BYTE *)v7;
    }
    else
    {
      v12 = (unsigned __int8)*v11;
    }
  }
  if ( (unsigned __int8)v12 <= 0x15u && ((unsigned __int8)(v12 - 12) <= 1u || sub_AC30F0((__int64)v11)) )
  {
LABEL_7:
    v16 = 1;
    v17 = *(_QWORD *)(*(_QWORD *)a3 + 8LL);
    a3[4] = 0;
    *(_QWORD *)a3 = v17;
    sub_28AAD10(a1, (_QWORD *)a2, v7, v8, v9, v10);
    return v16;
  }
  v19 = *(_QWORD *)(a1 + 40);
  v20 = *(_QWORD *)(v19 + 40);
  v21 = *(_DWORD *)(v19 + 56);
  if ( !v21 )
    return 0;
  v22 = (unsigned int)(v21 - 1);
  v23 = v22 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v24 = (__int64 *)(v20 + 16LL * v23);
  v25 = *v24;
  if ( a2 != *v24 )
  {
    v55 = 1;
    while ( v25 != -4096 )
    {
      v74 = v55 + 1;
      v23 = v22 & (v55 + v23);
      v24 = (__int64 *)(v20 + 16LL * v23);
      v25 = *v24;
      if ( a2 == *v24 )
        goto LABEL_12;
      v55 = v74;
    }
    return 0;
  }
LABEL_12:
  v26 = (_BYTE *)v24[1];
  if ( !v26 )
    return 0;
  v27 = sub_BD3990(*(unsigned __int8 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), v22);
  if ( *v27 == 3 )
  {
    v16 = v27[80] & 1;
    if ( (v27[80] & 1) != 0 )
    {
      v85 = (__int64)v27;
      if ( !sub_B2FC80((__int64)v27) && !(unsigned __int8)sub_B2F6B0(v85) && (*(_BYTE *)(v85 + 80) & 2) == 0 )
      {
        v56 = sub_B43CC0(a2);
        v57 = sub_98A180(*(unsigned __int8 **)(v85 - 32), v56);
        if ( v57 )
        {
          v86 = v57;
          sub_23D0AB0((__int64)&v92, a2, 0, 0, 0);
          LOWORD(v58) = sub_A74840((_QWORD *)(a2 + 72), 0);
          v59 = sub_B34240(
                  (__int64)&v92,
                  *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
                  v86,
                  *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
                  v58,
                  0,
                  0,
                  0,
                  0);
          v60 = (__int64 *)sub_D69570(*(_QWORD **)(a1 + 48), v59, 0, (__int64)v26);
          sub_D75120(*(__int64 **)(a1 + 48), v60, 1);
          sub_28AAD10(a1, (_QWORD *)a2, v61, v62, v63, v64);
          nullsub_61();
          v100 = &unk_49DA100;
          nullsub_63();
          if ( (__int64 *)v92.m128i_i64[0] != &v93 )
            _libc_free(v92.m128i_u64[0]);
          return v16;
        }
      }
    }
  }
  v28 = *(_QWORD *)(a1 + 8);
  v29 = *(_QWORD *)(a1 + 56);
  v93 = 0;
  v94 = 1;
  v92.m128i_i64[0] = v28;
  v92.m128i_i64[1] = v28;
  v30 = &v95;
  do
  {
    *v30 = -4;
    v30 += 5;
    *(v30 - 4) = -3;
    *(v30 - 3) = -4;
    *(v30 - 2) = -3;
  }
  while ( v30 != v101 );
  v101[0] = v29;
  v103 = v105;
  v104 = 0x400000000LL;
  v101[1] = 0;
  v102 = 0;
  v106 = 256;
  v108 = 0;
  v109 = 1;
  v107 = &unk_49DDBE8;
  v31 = &v110;
  do
  {
    *v31 = -4096;
    v31 += 2;
  }
  while ( v31 != (__int64 *)&v112 );
  v32 = (__m128i **)(v26 - 64);
  if ( *v26 == 26 )
    v32 = (__m128i **)(v26 - 32);
  v33 = *v32;
  sub_D67210(v88, a2);
  v34 = sub_103E0E0(*(_QWORD **)(a1 + 40));
  v35 = (*(__int64 (__fastcall **)(_QWORD *, __m128i *, __m128i *, __m128i *))(*v34 + 24LL))(v34, v33, v88, &v92);
  if ( *(_BYTE *)v35 == 27 )
  {
    v36 = *(_QWORD *)(v35 + 72);
    if ( v36 )
    {
      if ( *(_BYTE *)v36 == 85 )
      {
        v73 = *(_QWORD *)(v36 - 32);
        if ( v73 )
        {
          if ( !*(_BYTE *)v73
            && *(_QWORD *)(v73 + 24) == *(_QWORD *)(v36 + 80)
            && (*(_BYTE *)(v73 + 33) & 0x20) != 0
            && ((*(_DWORD *)(v73 + 36) - 243) & 0xFFFFFFFD) == 0
            && *(_QWORD *)(v35 + 64) == *(_QWORD *)(a2 + 40)
            && (unsigned __int8)sub_28AF6D0(a1, a2, v36, &v92) )
          {
            goto LABEL_65;
          }
        }
      }
    }
  }
  v75 = sub_103E0E0(*(_QWORD **)(a1 + 40));
  v78 = *(__int64 (__fastcall **)(_QWORD *, __m128i *, __m128i *, __m128i *))(*v75 + 24LL);
  sub_D671D0(&v89, a2);
  v37 = v33;
  v38 = v78(v75, v33, &v89, &v92);
  v39 = v38;
  if ( *(_BYTE *)v38 != 27 )
    goto LABEL_38;
  v40 = *(_QWORD *)(v38 + 72);
  v41 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v42 = 32 * (2 - v41);
  v43 = *(_BYTE **)(a2 + v42);
  if ( !v40 )
  {
LABEL_37:
    v83 = (__int64)v43;
    v53 = sub_BD3990(*(unsigned __int8 **)(a2 + 32 * (1 - v41)), (__int64)v37);
    v37 = &v92;
    if ( (unsigned __int8)sub_28AAA80(*(_QWORD *)(a1 + 40), v92.m128i_i64, v53, v39, v83) )
      goto LABEL_64;
LABEL_38:
    v54 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), (__int64)v37);
    if ( *v54 != 60 )
      goto LABEL_39;
    v66 = sub_BD3990(*(unsigned __int8 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), (__int64)v37);
    if ( *v66 != 60 )
      goto LABEL_39;
    v67 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    if ( *(_BYTE *)v67 != 17 )
      goto LABEL_39;
    v68 = *(_QWORD **)(v67 + 24);
    if ( *(_DWORD *)(v67 + 32) > 0x40u )
      v68 = (_QWORD *)*v68;
    v89.m128i_i64[0] = (__int64)v68;
    v89.m128i_i8[8] = 0;
    v16 = sub_28AC0B0(a1, a2, a2, (__int64)v54, (__int64)v66, (__int64)&v92, (unsigned __int64)v68, 0);
    if ( (_BYTE)v16 )
    {
      *(_QWORD *)a3 = sub_B46B10(a2, 0) + 24;
      a3[4] = 0;
      sub_28AAD10(a1, (_QWORD *)a2, v69, v70, v71, v72);
    }
    else
    {
LABEL_39:
      v16 = 0;
    }
    goto LABEL_40;
  }
  if ( *v43 == 17 )
  {
    if ( *(_BYTE *)v40 != 85 )
      goto LABEL_37;
    v76 = *(_QWORD *)(a2 + v42);
    v91 = sub_28A9430;
    v89.m128i_i64[0] = v40;
    v79 = v40;
    v90 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_28A9520;
    v44 = sub_A74840((_QWORD *)(a2 + 72), 0);
    v45 = 0;
    if ( HIBYTE(v44) )
      v45 = v44;
    v46 = *(_QWORD **)(v76 + 24);
    if ( *(_DWORD *)(v76 + 32) > 0x40u )
      v46 = (_QWORD *)*v46;
    v87 = (char)v46;
    v77 = v45;
    v81 = sub_BD3990(*(unsigned __int8 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), 0);
    v47 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), 0);
    v37 = (__m128i *)a2;
    v48 = sub_28AADB0((_QWORD *)a1, a2, a2, (__int64)v47, (__int64)v81, v77, v87, 0, &v92, (__int64)&v89);
    v40 = v79;
    v52 = v48;
    if ( v90 )
    {
      v80 = v48;
      v82 = v40;
      v37 = &v89;
      v90(&v89, &v89, 3);
      v52 = v80;
      v40 = v82;
    }
    if ( (_BYTE)v52 )
      goto LABEL_64;
  }
  if ( *(_BYTE *)v40 != 85 )
    goto LABEL_36;
  v65 = *(_QWORD *)(v40 - 32);
  if ( !v65 )
    goto LABEL_36;
  if ( !*(_BYTE *)v65
    && *(_QWORD *)(v65 + 24) == *(_QWORD *)(v40 + 80)
    && (*(_BYTE *)(v65 + 33) & 0x20) != 0
    && ((*(_DWORD *)(v65 + 36) - 238) & 0xFFFFFFFD) == 0 )
  {
    v37 = (__m128i *)a2;
    v84 = (_BYTE *)v40;
    if ( (unsigned __int8)sub_28ADBE0(a1, a2, v40, &v92) )
      goto LABEL_65;
    v40 = (__int64)v84;
    if ( *v84 != 85 )
      goto LABEL_36;
    v65 = *((_QWORD *)v84 - 4);
    if ( !v65 )
      goto LABEL_36;
  }
  if ( *(_BYTE *)v65
    || *(_QWORD *)(v65 + 24) != *(_QWORD *)(v40 + 80)
    || (*(_BYTE *)(v65 + 33) & 0x20) == 0
    || ((*(_DWORD *)(v65 + 36) - 243) & 0xFFFFFFFD) != 0
    || (v37 = (__m128i *)a2, !(unsigned __int8)sub_28AF7E0(a1, a2, v40, v92.m128i_i64)) )
  {
LABEL_36:
    v41 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    v43 = *(_BYTE **)(a2 + 32 * (2 - v41));
    goto LABEL_37;
  }
LABEL_64:
  sub_28AAD10(a1, (_QWORD *)a2, v49, v52, v50, v51);
LABEL_65:
  v16 = 1;
LABEL_40:
  v107 = &unk_49DDBE8;
  if ( (v109 & 1) == 0 )
    sub_C7D6A0(v110, 16LL * v111, 8);
  nullsub_184();
  if ( v103 != v105 )
    _libc_free((unsigned __int64)v103);
  if ( (v94 & 1) == 0 )
    sub_C7D6A0(v95, 40LL * (unsigned int)v96, 8);
  return v16;
}
