// Function: sub_28B0E80
// Address: 0x28b0e80
//
__int64 __fastcall sub_28B0E80(_QWORD *a1, __int64 a2, unsigned int a3, __m128i a4, __m128i a5, __m128i a6)
{
  _QWORD *v9; // rax
  __int64 v10; // rcx
  __int64 *v11; // rax
  __int64 *v12; // rax
  int v13; // edx
  __int64 v14; // r15
  __int16 v15; // ax
  unsigned int v16; // r13d
  __int64 v18; // rsi
  unsigned __int8 *v19; // rax
  unsigned __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rsi
  int v23; // eax
  int v24; // r10d
  unsigned __int8 *v25; // r8
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // rcx
  __int64 *v29; // rdi
  __int64 v30; // rax
  __int64 *v31; // rsi
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // r10
  unsigned __int8 *v36; // r8
  __int64 v37; // rdx
  __int64 v38; // rax
  int v39; // eax
  bool v40; // al
  __int64 v41; // r10
  __int64 v42; // r15
  unsigned __int64 *v43; // rax
  unsigned int v44; // ecx
  __int64 v45; // r8
  __int64 v46; // r10
  unsigned __int64 *v47; // rax
  unsigned __int16 v48; // ax
  unsigned __int64 *v49; // r10
  unsigned __int8 v50; // si
  unsigned __int64 v51; // rax
  char v52; // al
  unsigned __int8 v53; // r15
  __int64 v54; // r9
  __int64 v55; // r8
  unsigned __int64 *v56; // rdi
  unsigned int v57; // eax
  unsigned __int8 *v58; // rax
  unsigned __int8 v59; // al
  __int64 v60; // rdi
  __int64 v61; // r13
  __int64 v62; // r9
  _QWORD *v63; // r8
  __m128i *v64; // rsi
  __int64 *v65; // rdi
  __int64 i; // rcx
  __m128i *v67; // rdi
  __m128i *v68; // rsi
  __int64 j; // rcx
  unsigned __int64 *v70; // rsi
  unsigned __int8 *v71; // rax
  int v72; // eax
  int v73; // r9d
  int v74; // eax
  __int64 v75; // [rsp-30h] [rbp-450h] BYREF
  unsigned int v76; // [rsp+4h] [rbp-41Ch]
  __int64 v77; // [rsp+8h] [rbp-418h]
  __int64 v78; // [rsp+10h] [rbp-410h]
  _QWORD **v79; // [rsp+18h] [rbp-408h]
  unsigned __int8 *v80; // [rsp+20h] [rbp-400h]
  unsigned __int64 *v81; // [rsp+28h] [rbp-3F8h]
  unsigned __int8 *v82; // [rsp+30h] [rbp-3F0h]
  __int64 v83; // [rsp+38h] [rbp-3E8h]
  unsigned __int64 *v84; // [rsp+40h] [rbp-3E0h]
  __int64 v85; // [rsp+48h] [rbp-3D8h]
  unsigned __int64 v86; // [rsp+50h] [rbp-3D0h] BYREF
  unsigned __int8 v87; // [rsp+58h] [rbp-3C8h]
  char v88; // [rsp+60h] [rbp-3C0h]
  _QWORD v89[6]; // [rsp+70h] [rbp-3B0h] BYREF
  __m128i v90[3]; // [rsp+A0h] [rbp-380h] BYREF
  __m128i v91[3]; // [rsp+D0h] [rbp-350h] BYREF
  __m128i v92; // [rsp+100h] [rbp-320h] BYREF
  __int64 v93; // [rsp+110h] [rbp-310h]
  __int64 v94; // [rsp+118h] [rbp-308h]
  __int64 v95; // [rsp+120h] [rbp-300h]
  __int64 v96; // [rsp+128h] [rbp-2F8h]
  char v97; // [rsp+130h] [rbp-2F0h]
  _QWORD *v98; // [rsp+140h] [rbp-2E0h] BYREF
  _QWORD v99[2]; // [rsp+148h] [rbp-2D8h] BYREF
  __int64 v100; // [rsp+158h] [rbp-2C8h]
  __int64 v101; // [rsp+160h] [rbp-2C0h] BYREF
  unsigned int v102; // [rsp+168h] [rbp-2B8h]
  _QWORD v103[2]; // [rsp+2A0h] [rbp-180h] BYREF
  char v104; // [rsp+2B0h] [rbp-170h]
  _BYTE *v105; // [rsp+2B8h] [rbp-168h]
  __int64 v106; // [rsp+2C0h] [rbp-160h]
  _BYTE v107[128]; // [rsp+2C8h] [rbp-158h] BYREF
  __int16 v108; // [rsp+348h] [rbp-D8h]
  void *v109; // [rsp+350h] [rbp-D0h]
  __int64 v110; // [rsp+358h] [rbp-C8h]
  __int64 v111; // [rsp+360h] [rbp-C0h]
  __int64 v112; // [rsp+368h] [rbp-B8h] BYREF
  unsigned int v113; // [rsp+370h] [rbp-B0h]
  char v114; // [rsp+3E8h] [rbp-38h] BYREF

  v9 = (_QWORD *)a1[1];
  v10 = a1[7];
  v99[1] = 0;
  v100 = 1;
  v98 = v9;
  v99[0] = v9;
  v11 = &v101;
  do
  {
    *v11 = -4;
    v11 += 5;
    *(v11 - 4) = -3;
    *(v11 - 3) = -4;
    *(v11 - 2) = -3;
  }
  while ( v11 != v103 );
  v103[0] = v10;
  v106 = 0x400000000LL;
  v108 = 256;
  v103[1] = 0;
  v104 = 0;
  v105 = v107;
  v110 = 0;
  v111 = 1;
  v109 = &unk_49DDBE8;
  v12 = &v112;
  do
  {
    *v12 = -4096;
    v12 += 2;
  }
  while ( v12 != (__int64 *)&v114 );
  v13 = *(_DWORD *)(a2 + 4);
  v85 = a3;
  v14 = *(_QWORD *)(a2 + 32 * (a3 - (unsigned __int64)(v13 & 0x7FFFFFF)));
  v15 = sub_B49EE0((unsigned __int8 *)a2, a3);
  LOBYTE(v84) = v15 | HIBYTE(v15);
  if ( v15 )
    goto LABEL_6;
  v18 = a3;
  v16 = sub_B49B80(a2, a3, 22);
  if ( (_BYTE)v16 )
    goto LABEL_16;
  v18 = a2;
  v92.m128i_i64[0] = v14;
  v92.m128i_i64[1] = -1;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 1;
  if ( (sub_CF63E0(v98, (unsigned __int8 *)a2, &v92, (__int64)v99) & 2) == 0 )
  {
LABEL_16:
    v83 = sub_B43CC0(a2);
    v19 = sub_BD3990((unsigned __int8 *)v14, v18);
    if ( *v19 != 60 )
      goto LABEL_6;
    v82 = v19;
    v81 = &v86;
    sub_B4CED0((__int64)&v86, (__int64)v19, v83);
    if ( !v88 )
      goto LABEL_6;
    v16 = v87;
    if ( v87 )
      goto LABEL_6;
    v20 = v86;
    v21 = a1[5];
    v89[5] = 0;
    v89[0] = v14;
    v22 = *(_QWORD *)(v21 + 40);
    v89[2] = 0;
    v89[3] = 0;
    if ( v86 > 0x3FFFFFFFFFFFFFFBLL )
      v20 = 0xBFFFFFFFFFFFFFFELL;
    v89[4] = 0;
    v89[1] = v20;
    v23 = *(_DWORD *)(v21 + 56);
    if ( !v23 )
      goto LABEL_6;
    v24 = v23 - 1;
    v25 = v82;
    v26 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v27 = (__int64 *)(v22 + 16LL * v26);
    v28 = *v27;
    if ( *v27 != a2 )
    {
      v72 = 1;
      while ( v28 != -4096 )
      {
        v73 = v72 + 1;
        v26 = v24 & (v72 + v26);
        v27 = (__int64 *)(v22 + 16LL * v26);
        v28 = *v27;
        if ( *v27 == a2 )
          goto LABEL_23;
        v72 = v73;
      }
      goto LABEL_6;
    }
LABEL_23:
    v82 = (unsigned __int8 *)v27[1];
    if ( !v82 )
      goto LABEL_6;
    v80 = v25;
    v29 = sub_103E0E0((_QWORD *)v21);
    v30 = *v29;
    v31 = (__int64 *)(v82 - 64);
    if ( *v82 == 26 )
      v31 = (__int64 *)(v82 - 32);
    v79 = &v98;
    v32 = *v31;
    v33 = (*(__int64 (__fastcall **)(__int64 *, __int64, _QWORD *))(v30 + 24))(v29, v32, v89);
    if ( *(_BYTE *)v33 != 27 )
      goto LABEL_6;
    v34 = sub_28AD070(*(_QWORD *)(v33 + 72));
    v35 = v34;
    if ( !v34 )
      goto LABEL_6;
    v36 = v80;
    v37 = *(_DWORD *)(v34 + 4) & 0x7FFFFFF;
    v38 = *(_QWORD *)(v34 + 32 * (3 - v37));
    if ( *(_DWORD *)(v38 + 32) <= 0x40u )
    {
      v40 = *(_QWORD *)(v38 + 24) == 0;
    }
    else
    {
      v76 = *(_DWORD *)(v38 + 32);
      v77 = v37;
      v78 = v35;
      v39 = sub_C444A0(v38 + 24);
      v36 = v80;
      v35 = v78;
      v37 = v77;
      v40 = v76 == v39;
    }
    if ( !v40 )
      goto LABEL_6;
    v78 = (__int64)v36;
    v80 = (unsigned __int8 *)v35;
    if ( v36 != sub_BD3990(*(unsigned __int8 **)(v35 - 32 * v37), v32) )
      goto LABEL_6;
    if ( *(_QWORD *)(v14 + 8) != *((_QWORD *)sub_BD3990(
                                               *(unsigned __int8 **)&v80[32 * (1LL - (*((_DWORD *)v80 + 1) & 0x7FFFFFF))],
                                               v32)
                                 + 1) )
      goto LABEL_6;
    v41 = (__int64)v80;
    v42 = *(_QWORD *)&v80[32 * (2LL - (*((_DWORD *)v80 + 1) & 0x7FFFFFF))];
    if ( *(_BYTE *)v42 != 17 )
      goto LABEL_6;
    LOBYTE(v80) = v88;
    if ( !v88 )
      goto LABEL_6;
    v77 = v41;
    v43 = (unsigned __int64 *)sub_CA1930(v81);
    v44 = *(_DWORD *)(v42 + 32);
    v45 = v78;
    v81 = v43;
    v46 = v77;
    v76 = v44;
    if ( v44 > 0x40 )
    {
      v74 = sub_C444A0(v42 + 24);
      v45 = v78;
      v46 = v77;
      if ( v76 - v74 > 0x40 )
        goto LABEL_6;
      v47 = **(unsigned __int64 ***)(v42 + 24);
    }
    else
    {
      v47 = *(unsigned __int64 **)(v42 + 24);
    }
    if ( v81 != v47 )
    {
LABEL_6:
      v16 = 0;
      goto LABEL_7;
    }
    v78 = v45;
    v81 = (unsigned __int64 *)v46;
    v48 = sub_A74840((_QWORD *)(v46 + 72), 1);
    v49 = v81;
    v50 = (unsigned __int8)v84;
    if ( HIBYTE(v48) )
      v50 = v48;
    _BitScanReverse64(&v51, 1LL << *(_WORD *)(v78 + 2));
    v52 = v51 ^ 0x3F;
    v53 = 63 - v52;
    if ( (unsigned __int8)(63 - v52) <= v50 )
      goto LABEL_42;
    v54 = a1[3];
    v55 = a1[2];
    v56 = v81;
    v84 = v81;
    v57 = (unsigned __int8)(63 - v52);
    v77 = v54;
    BYTE1(v57) = 1;
    v78 = v55;
    LODWORD(v81) = v57;
    v58 = sub_28B06A0((__int64)v56, v57);
    v59 = sub_F518D0(v58, (unsigned int)v81, v83, a2, v78, v77);
    v49 = v84;
    if ( v59 >= v53 )
    {
LABEL_42:
      v60 = a1[5];
      v84 = v49;
      v61 = sub_28AACA0(v60, (__int64)v49);
      sub_D671D0(v91, (__int64)v84);
      v63 = (_QWORD *)a1[5];
      v64 = v91;
      v65 = &v75;
      for ( i = 12; i; --i )
      {
        *(_DWORD *)v65 = v64->m128i_i32[0];
        v64 = (__m128i *)((char *)v64 + 4);
        v65 = (__int64 *)((char *)v65 + 4);
      }
      if ( !sub_28A97D0(v63, v79, v61, (__int64)v82, (__int64)v63, v62, a4, a5, a6) )
      {
        sub_D671D0(v90, (__int64)v84);
        v67 = &v92;
        v68 = v90;
        for ( j = 12; j; --j )
        {
          v67->m128i_i32[0] = v68->m128i_i32[0];
          v68 = (__m128i *)((char *)v68 + 4);
          v67 = (__m128i *)((char *)v67 + 4);
        }
        v97 = 1;
        if ( (sub_CF63E0(v98, (unsigned __int8 *)a2, &v92, (__int64)v99) & 2) == 0 )
        {
          v70 = v84;
          sub_F57040((_BYTE *)a2, (__int64)v84);
          v71 = sub_28B06A0((__int64)v84, (__int64)v70);
          sub_AC2B30(a2 + 32 * (v85 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), (__int64)v71);
          v16 = (unsigned __int8)v80;
          goto LABEL_7;
        }
      }
      goto LABEL_6;
    }
  }
LABEL_7:
  v109 = &unk_49DDBE8;
  if ( (v111 & 1) == 0 )
    sub_C7D6A0(v112, 16LL * v113, 8);
  nullsub_184();
  if ( v105 != v107 )
    _libc_free((unsigned __int64)v105);
  if ( (v100 & 1) == 0 )
    sub_C7D6A0(v101, 40LL * v102, 8);
  return v16;
}
