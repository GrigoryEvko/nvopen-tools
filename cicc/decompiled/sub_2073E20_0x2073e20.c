// Function: sub_2073E20
// Address: 0x2073e20
//
void __fastcall sub_2073E20(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v7; // rdx
  int v8; // eax
  __int64 v9; // rsi
  int v10; // edx
  unsigned int v11; // eax
  __int64 v12; // rcx
  int v13; // edi
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r15
  __int64 v18; // r12
  unsigned int v19; // eax
  __int64 v20; // r8
  __int64 v21; // rsi
  unsigned __int64 v22; // r13
  __int64 v23; // rax
  unsigned __int64 v24; // rax
  unsigned int v25; // r12d
  __int64 v26; // r13
  unsigned int v27; // eax
  __m128i v28; // rax
  __int64 v29; // r15
  unsigned int v30; // r12d
  __int64 v31; // rax
  unsigned int v32; // edx
  unsigned __int8 v33; // al
  __int64 v34; // r10
  unsigned int v35; // r12d
  __int64 v36; // rdx
  __int128 v37; // rax
  __int64 *v38; // rax
  unsigned int v39; // edx
  unsigned int v40; // r10d
  __int64 v41; // r13
  _QWORD *v42; // rdx
  __int64 *v43; // r15
  __int64 v44; // rdi
  __int64 v45; // rdx
  __m128i v46; // xmm0
  __int64 v47; // rdi
  __int64 v48; // rdx
  const void ***v49; // rax
  int v50; // edx
  __int64 v51; // r9
  __int64 *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r13
  __int64 *v55; // r12
  __int64 *v56; // rax
  __int64 v57; // r13
  unsigned int v58; // esi
  int v59; // eax
  __int64 v60; // rdi
  __int64 v61; // rax
  _QWORD *v62; // rax
  __int64 v63; // rax
  unsigned int v64; // edx
  __int64 (*v65)(); // rax
  __int64 v66; // rax
  unsigned int v67; // edx
  __int64 *v68; // r13
  unsigned int v69; // eax
  __int128 v70; // rax
  const void **v71; // r8
  __int64 v72; // rcx
  __int64 *v73; // rax
  __int64 *v74; // r15
  unsigned int v75; // edx
  unsigned int v76; // r13d
  __int128 v77; // rax
  const void **v78; // r8
  __int64 v79; // rcx
  unsigned int v80; // edx
  __int128 v81; // [rsp-20h] [rbp-100h]
  unsigned int v82; // [rsp+0h] [rbp-E0h]
  __int64 v83; // [rsp+0h] [rbp-E0h]
  __int64 v84; // [rsp+0h] [rbp-E0h]
  __int64 v85; // [rsp+8h] [rbp-D8h]
  __int64 v86; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v87; // [rsp+8h] [rbp-D8h]
  unsigned int v88; // [rsp+8h] [rbp-D8h]
  __int64 v89; // [rsp+8h] [rbp-D8h]
  unsigned __int32 v90; // [rsp+10h] [rbp-D0h]
  __int64 *v91; // [rsp+10h] [rbp-D0h]
  __int64 v92; // [rsp+10h] [rbp-D0h]
  __int64 v93; // [rsp+10h] [rbp-D0h]
  __m128i v94; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v95; // [rsp+30h] [rbp-B0h]
  __int64 v96; // [rsp+38h] [rbp-A8h]
  __int64 *v97; // [rsp+40h] [rbp-A0h]
  __int64 v98; // [rsp+48h] [rbp-98h]
  __int64 *v99; // [rsp+50h] [rbp-90h]
  __int64 v100; // [rsp+58h] [rbp-88h]
  __int64 v101; // [rsp+68h] [rbp-78h] BYREF
  __int64 v102; // [rsp+70h] [rbp-70h] BYREF
  int v103; // [rsp+78h] [rbp-68h]
  _QWORD v104[2]; // [rsp+80h] [rbp-60h] BYREF
  __m128i v105; // [rsp+90h] [rbp-50h]
  __int64 v106; // [rsp+A0h] [rbp-40h]
  __int64 v107; // [rsp+A8h] [rbp-38h]

  v7 = *(_QWORD *)(a1 + 712);
  v8 = *(_DWORD *)(v7 + 360);
  if ( v8 )
  {
    v9 = *(_QWORD *)(v7 + 344);
    v10 = v8 - 1;
    v11 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v12 = *(_QWORD *)(v9 + 16LL * v11);
    if ( a2 == v12 )
      return;
    v13 = 1;
    while ( v12 != -8 )
    {
      v11 = v10 & (v13 + v11);
      v12 = *(_QWORD *)(v9 + 16LL * v11);
      if ( a2 == v12 )
        return;
      ++v13;
    }
  }
  v14 = *(_DWORD *)(a1 + 536);
  v15 = *(_QWORD *)a1;
  v102 = 0;
  v103 = v14;
  if ( v15 )
  {
    if ( &v102 != (__int64 *)(v15 + 48) )
    {
      v16 = *(_QWORD *)(v15 + 48);
      v102 = v16;
      if ( v16 )
        sub_1623A60((__int64)&v102, v16, 2);
    }
  }
  v17 = *(_QWORD *)(a2 + 56);
  v18 = 1;
  v94.m128i_i64[0] = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
  v19 = sub_15A9FE0(v94.m128i_i64[0], v17);
  v20 = v94.m128i_i64[0];
  v21 = v17;
  v22 = v19;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v21 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v63 = *(_QWORD *)(v21 + 32);
        v21 = *(_QWORD *)(v21 + 24);
        v18 *= v63;
        continue;
      case 1:
        v23 = 16;
        goto LABEL_13;
      case 2:
        v23 = 32;
        goto LABEL_13;
      case 3:
      case 9:
        v23 = 64;
        goto LABEL_13;
      case 4:
        v23 = 80;
        goto LABEL_13;
      case 5:
      case 6:
        v23 = 128;
        goto LABEL_13;
      case 7:
        v58 = 0;
        goto LABEL_30;
      case 0xB:
        v23 = *(_DWORD *)(v21 + 8) >> 8;
        goto LABEL_13;
      case 0xD:
        v62 = (_QWORD *)sub_15A9930(v94.m128i_i64[0], v21);
        v20 = v94.m128i_i64[0];
        v23 = 8LL * *v62;
        goto LABEL_13;
      case 0xE:
        v60 = v94.m128i_i64[0];
        v93 = v94.m128i_i64[0];
        v83 = *(_QWORD *)(v21 + 24);
        v94.m128i_i64[0] = *(_QWORD *)(v21 + 32);
        v87 = (unsigned int)sub_15A9FE0(v60, v83);
        v61 = sub_127FA20(v93, v83);
        v20 = v93;
        v23 = 8 * v94.m128i_i64[0] * v87 * ((v87 + ((unsigned __int64)(v61 + 7) >> 3) - 1) / v87);
        goto LABEL_13;
      case 0xF:
        v58 = *(_DWORD *)(v21 + 8) >> 8;
LABEL_30:
        v59 = sub_15A9520(v94.m128i_i64[0], v58);
        v20 = v94.m128i_i64[0];
        v23 = (unsigned int)(8 * v59);
LABEL_13:
        v85 = v20;
        v24 = (v22 + ((unsigned __int64)(v23 * v18 + 7) >> 3) - 1) / v22;
        v25 = (unsigned int)(1 << *(_WORD *)(a2 + 18)) >> 1;
        v26 = v24 * v22;
        v27 = sub_15AAE50(v20, v17);
        if ( v25 >= v27 )
          v27 = v25;
        v82 = v27;
        v28.m128i_i64[0] = (__int64)sub_20685E0(a1, *(__int64 **)(a2 - 24), a3, a4, a5);
        v94 = v28;
        v29 = v28.m128i_i64[0];
        v30 = *(_DWORD *)(v85 + 4);
        v90 = v28.m128i_u32[2];
        v31 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
        v32 = 8 * sub_15A9520(v31, v30);
        if ( v32 == 32 )
        {
          v33 = 5;
        }
        else if ( v32 > 0x20 )
        {
          v33 = 6;
          if ( v32 != 64 )
          {
            v33 = 0;
            if ( v32 == 128 )
              v33 = 7;
          }
        }
        else
        {
          v33 = 3;
          if ( v32 != 8 )
            v33 = 4 * (v32 == 16);
        }
        v34 = v90;
        v35 = v33;
        v36 = *(_QWORD *)(v29 + 40) + 16LL * v90;
        if ( *(_BYTE *)v36 != v33 || !v33 && *(_QWORD *)(v36 + 8) )
        {
          v29 = sub_1D323C0(
                  *(__int64 **)(a1 + 552),
                  v94.m128i_i64[0],
                  v94.m128i_i64[1],
                  (__int64)&v102,
                  v33,
                  0,
                  *(double *)a3.m128i_i64,
                  *(double *)a4.m128i_i64,
                  *(double *)a5.m128i_i64);
          v34 = v64;
        }
        v86 = v34;
        v91 = *(__int64 **)(a1 + 552);
        *(_QWORD *)&v37 = sub_1D38BB0((__int64)v91, v26, (__int64)&v102, v35, 0, 0, a3, *(double *)a4.m128i_i64, a5, 0);
        v94.m128i_i64[0] = v29;
        v94.m128i_i64[1] = v86 | v94.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        v38 = sub_1D332F0(
                v91,
                54,
                (__int64)&v102,
                v35,
                0,
                0,
                *(double *)a3.m128i_i64,
                *(double *)a4.m128i_i64,
                a5,
                v29,
                v94.m128i_u64[1],
                v37);
        v40 = v39;
        v41 = v39;
        v42 = *(_QWORD **)(a1 + 552);
        v43 = v38;
        if ( (unsigned int)(*(_DWORD *)(*v42 + 504LL) - 34) > 1 )
        {
          v88 = v40;
          v65 = *(__int64 (**)())(**(_QWORD **)(v42[4] + 16LL) + 48LL);
          if ( v65 == sub_1D90020 )
            BUG();
          v66 = v65();
          v67 = 0;
          v68 = *(__int64 **)(a1 + 552);
          v69 = *(_DWORD *)(v66 + 12);
          if ( v82 > v69 )
            v67 = v82;
          v84 = v69 - 1;
          v92 = v67;
          *(_QWORD *)&v70 = sub_1D38BB0(
                              *(_QWORD *)(a1 + 552),
                              v84,
                              (__int64)&v102,
                              v35,
                              0,
                              0,
                              a3,
                              *(double *)a4.m128i_i64,
                              a5,
                              0);
          v94.m128i_i64[0] = (__int64)v43;
          v71 = *(const void ***)(v43[5] + 16LL * v88 + 8);
          v72 = *(unsigned __int8 *)(v43[5] + 16LL * v88);
          v94.m128i_i64[1] = v88 | v94.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          v73 = sub_1D332F0(
                  v68,
                  52,
                  (__int64)&v102,
                  v72,
                  v71,
                  3u,
                  *(double *)a3.m128i_i64,
                  *(double *)a4.m128i_i64,
                  a5,
                  (__int64)v43,
                  v94.m128i_u64[1],
                  v70);
          v74 = *(__int64 **)(a1 + 552);
          v76 = v75;
          v89 = (__int64)v73;
          *(_QWORD *)&v77 = sub_1D38BB0(
                              (__int64)v74,
                              ~v84,
                              (__int64)&v102,
                              v35,
                              0,
                              0,
                              a3,
                              *(double *)a4.m128i_i64,
                              a5,
                              0);
          v94.m128i_i64[0] = v89;
          v78 = *(const void ***)(*(_QWORD *)(v89 + 40) + 16LL * v76 + 8);
          v79 = *(unsigned __int8 *)(*(_QWORD *)(v89 + 40) + 16LL * v76);
          v94.m128i_i64[1] = v76 | v94.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          v43 = sub_1D332F0(
                  v74,
                  118,
                  (__int64)&v102,
                  v79,
                  v78,
                  0,
                  *(double *)a3.m128i_i64,
                  *(double *)a4.m128i_i64,
                  a5,
                  v89,
                  v94.m128i_u64[1],
                  v77);
          v41 = v80;
        }
        else
        {
          v92 = v82;
        }
        v104[0] = sub_2051C20((__int64 *)a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
        v44 = *(_QWORD *)(a1 + 552);
        v104[1] = v45;
        v94.m128i_i64[0] = (__int64)v43;
        v94.m128i_i64[1] = v41 | v94.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        v46 = _mm_load_si128(&v94);
        v105 = v46;
        v106 = sub_1D38BB0(v44, v92, (__int64)&v102, v35, 0, 0, v46, *(double *)a4.m128i_i64, a5, 0);
        v47 = *(_QWORD *)(a1 + 552);
        v107 = v48;
        v49 = (const void ***)sub_1D252B0(
                                v47,
                                *(unsigned __int8 *)(v43[5] + 16 * v41),
                                *(_QWORD *)(v43[5] + 16 * v41 + 8),
                                1,
                                0);
        *((_QWORD *)&v81 + 1) = 3;
        *(_QWORD *)&v81 = v104;
        v52 = sub_1D36D80(
                *(__int64 **)(a1 + 552),
                187,
                (__int64)&v102,
                v49,
                v50,
                *(double *)v46.m128i_i64,
                *(double *)a4.m128i_i64,
                a5,
                v51,
                v81);
        v54 = v53;
        v55 = v52;
        v101 = a2;
        v56 = sub_205F5C0(a1 + 8, &v101);
        v100 = v54;
        v99 = v55;
        v56[1] = (__int64)v55;
        *((_DWORD *)v56 + 4) = v100;
        v57 = *(_QWORD *)(a1 + 552);
        if ( v55 )
        {
          nullsub_686();
          v98 = 1;
          v97 = v55;
          *(_QWORD *)(v57 + 176) = v55;
          *(_DWORD *)(v57 + 184) = v98;
          sub_1D23870();
        }
        else
        {
          v96 = 1;
          v95 = 0;
          *(_QWORD *)(v57 + 176) = 0;
          *(_DWORD *)(v57 + 184) = v96;
        }
        if ( v102 )
          sub_161E7C0((__int64)&v102, v102);
        break;
    }
    break;
  }
}
