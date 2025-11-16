// Function: sub_2941D40
// Address: 0x2941d40
//
__int64 __fastcall sub_2941D40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 result; // rax
  __int64 v8; // r12
  __m128i v9; // xmm5
  __m128i v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r12
  unsigned __int64 v15; // rdi
  __int64 v16; // rdx
  int v17; // eax
  _QWORD *v18; // rdi
  __int64 v19; // r14
  char v20; // al
  __int64 v21; // r12
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r9
  unsigned __int64 v25; // rbx
  unsigned __int64 *v26; // rax
  unsigned __int64 *v27; // rcx
  unsigned __int64 *i; // rcx
  unsigned int v29; // eax
  __int64 *v30; // rbx
  _OWORD *v31; // rdx
  char v32; // al
  __int64 (__fastcall *v33)(__int64, _BYTE *, __int64, __int64); // rax
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // r15
  unsigned __int64 *v37; // r12
  __int64 v38; // rax
  __int64 v39; // r12
  __m128i v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // r14
  __int64 v45; // r12
  __int64 v46; // rdx
  unsigned int v47; // esi
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __m128i v51; // xmm1
  __m128i v52; // rax
  __int64 v53; // r8
  __int64 v54; // r9
  _BYTE *v55; // r13
  __m128i v56; // rax
  __int64 (__fastcall *v57)(__int64, _BYTE *, _BYTE *, __int64, __int64); // rax
  _QWORD *v58; // rax
  __int64 v59; // r15
  __int64 v60; // r13
  __int64 v61; // r12
  __int64 v62; // rdx
  unsigned int v63; // esi
  __m128i v64; // xmm3
  __int64 v65; // rbx
  unsigned __int64 *v66; // r13
  unsigned __int64 *v67; // r12
  unsigned __int64 *v68; // rdi
  int v69; // r13d
  unsigned __int64 v70; // rdi
  char v71; // dl
  char v72; // dh
  __int64 v73; // r8
  char v74; // al
  __int64 v75; // rcx
  char v76; // dl
  char v77; // dh
  __int64 v78; // r8
  char v79; // al
  __int64 v80; // rcx
  __int64 v81; // [rsp+8h] [rbp-378h]
  __int64 v82; // [rsp+10h] [rbp-370h]
  __int64 v83; // [rsp+20h] [rbp-360h]
  __int64 *v84; // [rsp+30h] [rbp-350h]
  __int64 *v85; // [rsp+58h] [rbp-328h]
  __int64 v86; // [rsp+60h] [rbp-320h]
  unsigned int v87; // [rsp+6Ch] [rbp-314h]
  __int64 *v88; // [rsp+70h] [rbp-310h]
  __int64 v90; // [rsp+88h] [rbp-2F8h]
  __int64 v91; // [rsp+90h] [rbp-2F0h]
  __int64 v92; // [rsp+98h] [rbp-2E8h]
  unsigned int v93; // [rsp+ACh] [rbp-2D4h] BYREF
  __m128i v94; // [rsp+B0h] [rbp-2D0h] BYREF
  __m128i v95; // [rsp+C0h] [rbp-2C0h] BYREF
  __int64 v96; // [rsp+D0h] [rbp-2B0h]
  _OWORD v97[2]; // [rsp+E0h] [rbp-2A0h] BYREF
  __int16 v98; // [rsp+100h] [rbp-280h]
  __m128i v99; // [rsp+110h] [rbp-270h] BYREF
  __m128i v100; // [rsp+120h] [rbp-260h]
  __int64 v101; // [rsp+130h] [rbp-250h]
  __m128i v102; // [rsp+140h] [rbp-240h] BYREF
  __m128i v103; // [rsp+150h] [rbp-230h] BYREF
  __int64 v104; // [rsp+160h] [rbp-220h]
  __m128i v105; // [rsp+170h] [rbp-210h] BYREF
  __m128i v106; // [rsp+180h] [rbp-200h] BYREF
  __int64 v107; // [rsp+190h] [rbp-1F0h]
  __int64 v108; // [rsp+1A8h] [rbp-1D8h]
  __int64 v109; // [rsp+1B0h] [rbp-1D0h]
  __int64 v110; // [rsp+1C0h] [rbp-1C0h]
  __int64 v111; // [rsp+1C8h] [rbp-1B8h]
  void *v112; // [rsp+1F0h] [rbp-190h]
  unsigned __int64 *v113; // [rsp+200h] [rbp-180h] BYREF
  __int64 v114; // [rsp+208h] [rbp-178h]
  _QWORD v115[15]; // [rsp+210h] [rbp-170h] BYREF

  v6 = *(unsigned int *)(a1 + 56);
  if ( (_DWORD)v6 )
  {
    v88 = *(__int64 **)(a1 + 48);
    v84 = &v88[2 * v6];
    v83 = a1 + 328;
    while ( 1 )
    {
      v19 = *v88;
      v85 = (__int64 *)v88[1];
      if ( !*(_QWORD *)(*v88 + 16) )
        goto LABEL_14;
      v86 = *(_QWORD *)(v19 + 8);
      v20 = *(_BYTE *)(v86 + 8);
      if ( v20 == 17 )
        break;
      if ( v20 == 15 )
      {
        v21 = *(_QWORD *)(v19 + 40);
        sub_23D0AB0((__int64)&v105, v19, 0, 0, 0);
        if ( *(_BYTE *)v19 == 84 )
        {
          v78 = sub_AA5190(v21);
          if ( v78 )
          {
            v79 = v77;
          }
          else
          {
            v79 = 0;
            v76 = 0;
          }
          v80 = v81;
          LOBYTE(v80) = v76;
          BYTE1(v80) = v79;
          v81 = v80;
          sub_A88F30((__int64)&v105, v21, v78, v80);
        }
        v25 = *(unsigned int *)(v86 + 12);
        v26 = v115;
        v114 = 0x400000000LL;
        v27 = v115;
        v87 = v25;
        v113 = v115;
        if ( v25 )
        {
          if ( v25 > 4 )
          {
            sub_2941C30((__int64)&v113, v25, v22, (__int64)v115, v23, v24);
            v27 = v113;
            v26 = &v113[10 * (unsigned int)v114];
          }
          for ( i = &v27[10 * v25]; i != v26; v26 += 10 )
          {
            if ( v26 )
            {
              *((_DWORD *)v26 + 2) = 0;
              *v26 = (unsigned __int64)(v26 + 2);
              *((_DWORD *)v26 + 3) = 8;
            }
          }
          v93 = 0;
          v90 = v19;
          LODWORD(v114) = v25;
          v29 = 0;
          while ( 2 )
          {
            v30 = (__int64 *)*v85;
            v91 = *v85 + 8LL * *((unsigned int *)v85 + 2);
            if ( v91 == *v85 )
            {
LABEL_60:
              v93 = ++v29;
              if ( v29 < v87 )
                continue;
              v19 = v90;
              v48 = sub_ACADE0((__int64 **)v86);
              v94.m128i_i32[0] = 0;
              v14 = v48;
              v49 = 0;
              while ( 2 )
              {
                sub_2939E80((__int64)&v102, a1, *(_QWORD *)(*(_QWORD *)(v86 + 16) + 8 * v49));
                v51 = _mm_loadu_si128(&v103);
                v97[0] = _mm_loadu_si128(&v102);
                v97[1] = v51;
                v52.m128i_i64[0] = (__int64)sub_BD5D20(v90);
                v102 = v52;
                LOWORD(v104) = 261;
                v55 = sub_293ACB0(
                        v105.m128i_i64,
                        v113[10 * v94.m128i_u32[0]],
                        (__int64)v97,
                        v52.m128i_i64[0],
                        v53,
                        v54,
                        (__int64 *)v52.m128i_i64[0],
                        v52.m128i_i64[1],
                        v103.m128i_i32[0],
                        v103.m128i_i32[2],
                        261);
                v56.m128i_i64[0] = (__int64)sub_BD5D20(v90);
                LOWORD(v101) = 773;
                v99 = v56;
                v100.m128i_i64[0] = (__int64)".insert";
                v57 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v110 + 88LL);
                if ( v57 == sub_9482E0 )
                {
                  if ( *(_BYTE *)v14 <= 0x15u && *v55 <= 0x15u )
                  {
                    v50 = sub_AAAE30(v14, (__int64)v55, &v94, 1);
                    goto LABEL_64;
                  }
                  goto LABEL_69;
                }
                v50 = v57(v110, (_BYTE *)v14, v55, (__int64)&v94, 1);
LABEL_64:
                if ( v50 )
                {
                  v14 = v50;
                }
                else
                {
LABEL_69:
                  LOWORD(v104) = 257;
                  v58 = sub_BD2C40(104, unk_3F148BC);
                  v59 = (__int64)v58;
                  if ( v58 )
                  {
                    sub_B44260((__int64)v58, *(_QWORD *)(v14 + 8), 65, 2u, 0, 0);
                    *(_QWORD *)(v59 + 72) = v59 + 88;
                    *(_QWORD *)(v59 + 80) = 0x400000000LL;
                    sub_B4FD20(v59, v14, (__int64)v55, &v94, 1, (__int64)&v102);
                  }
                  (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v111 + 16LL))(
                    v111,
                    v59,
                    &v99,
                    v108,
                    v109);
                  v60 = v105.m128i_i64[0];
                  v61 = v105.m128i_i64[0] + 16LL * v105.m128i_u32[2];
                  if ( v105.m128i_i64[0] != v61 )
                  {
                    do
                    {
                      v62 = *(_QWORD *)(v60 + 8);
                      v63 = *(_DWORD *)v60;
                      v60 += 16;
                      sub_B99FD0(v59, v63, v62);
                    }
                    while ( v61 != v60 );
                  }
                  v14 = v59;
                }
                v49 = (unsigned int)(v94.m128i_i32[0] + 1);
                v94.m128i_i32[0] = v49;
                if ( (unsigned int)v49 >= v87 )
                  goto LABEL_78;
                continue;
              }
            }
            break;
          }
          while ( 2 )
          {
            v39 = *v30;
            LODWORD(v97[0]) = v29;
            v98 = 265;
            v40.m128i_i64[0] = (__int64)sub_BD5D20(v90);
            v94 = v40;
            v95.m128i_i64[0] = (__int64)".elem";
            v32 = v98;
            LOWORD(v96) = 773;
            if ( (_BYTE)v98 )
            {
              if ( (_BYTE)v98 == 1 )
              {
                v64 = _mm_loadu_si128(&v95);
                v99 = _mm_loadu_si128(&v94);
                v101 = v96;
                v100 = v64;
              }
              else
              {
                if ( HIBYTE(v98) == 1 )
                {
                  v92 = *((_QWORD *)&v97[0] + 1);
                  v31 = *(_OWORD **)&v97[0];
                }
                else
                {
                  v31 = v97;
                  v32 = 2;
                }
                v100.m128i_i64[0] = (__int64)v31;
                LOBYTE(v101) = 2;
                v99.m128i_i64[0] = (__int64)&v94;
                v100.m128i_i64[1] = v92;
                BYTE1(v101) = v32;
              }
            }
            else
            {
              LOWORD(v101) = 256;
            }
            v33 = *(__int64 (__fastcall **)(__int64, _BYTE *, __int64, __int64))(*(_QWORD *)v110 + 80LL);
            if ( v33 == sub_92FAE0 )
            {
              if ( *(_BYTE *)v39 <= 0x15u )
              {
                v36 = sub_AAADB0(v39, &v93, 1);
                goto LABEL_43;
              }
              goto LABEL_49;
            }
            v36 = v33(v110, (_BYTE *)v39, (__int64)&v93, 1);
LABEL_43:
            if ( !v36 )
            {
LABEL_49:
              LOWORD(v104) = 257;
              v36 = (__int64)sub_BD2C40(104, 1u);
              if ( v36 )
              {
                v41 = sub_B501B0(*(_QWORD *)(v39 + 8), &v93, 1);
                sub_B44260(v36, v41, 64, 1u, 0, 0);
                if ( *(_QWORD *)(v36 - 32) )
                {
                  v42 = *(_QWORD *)(v36 - 24);
                  **(_QWORD **)(v36 - 16) = v42;
                  if ( v42 )
                    *(_QWORD *)(v42 + 16) = *(_QWORD *)(v36 - 16);
                }
                *(_QWORD *)(v36 - 32) = v39;
                v43 = *(_QWORD *)(v39 + 16);
                *(_QWORD *)(v36 - 24) = v43;
                if ( v43 )
                  *(_QWORD *)(v43 + 16) = v36 - 24;
                *(_QWORD *)(v36 - 16) = v39 + 16;
                *(_QWORD *)(v39 + 16) = v36 - 32;
                *(_QWORD *)(v36 + 72) = v36 + 88;
                *(_QWORD *)(v36 + 80) = 0x400000000LL;
                sub_B50030(v36, &v93, 1, (__int64)&v102);
              }
              (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v111 + 16LL))(
                v111,
                v36,
                &v99,
                v108,
                v109);
              v44 = v105.m128i_i64[0];
              v45 = v105.m128i_i64[0] + 16LL * v105.m128i_u32[2];
              if ( v105.m128i_i64[0] != v45 )
              {
                do
                {
                  v46 = *(_QWORD *)(v44 + 8);
                  v47 = *(_DWORD *)v44;
                  v44 += 16;
                  sub_B99FD0(v36, v47, v46);
                }
                while ( v45 != v44 );
              }
            }
            v37 = &v113[10 * v93];
            v38 = *((unsigned int *)v37 + 2);
            if ( v38 + 1 > (unsigned __int64)*((unsigned int *)v37 + 3) )
            {
              sub_C8D5F0((__int64)&v113[10 * v93], v37 + 2, v38 + 1, 8u, v34, v35);
              v38 = *((unsigned int *)v37 + 2);
            }
            ++v30;
            *(_QWORD *)(*v37 + 8 * v38) = v36;
            v29 = v93;
            ++*((_DWORD *)v37 + 2);
            if ( (__int64 *)v91 == v30 )
              goto LABEL_60;
            continue;
          }
        }
        v14 = sub_ACADE0((__int64 **)v86);
LABEL_78:
        v65 = (__int64)v113;
        v66 = &v113[10 * (unsigned int)v114];
        if ( v113 != v66 )
        {
          do
          {
            v66 -= 10;
            if ( (unsigned __int64 *)*v66 != v66 + 2 )
              _libc_free(*v66);
          }
          while ( (unsigned __int64 *)v65 != v66 );
          v66 = v113;
        }
        if ( v66 != v115 )
          _libc_free((unsigned __int64)v66);
        nullsub_61();
        v112 = &unk_49DA100;
        nullsub_63();
        v15 = v105.m128i_i64[0];
        if ( (__m128i *)v105.m128i_i64[0] == &v106 )
          goto LABEL_13;
LABEL_12:
        _libc_free(v15);
LABEL_13:
        sub_BD84D0(v19, v14);
LABEL_14:
        v16 = *(unsigned int *)(a1 + 336);
        v17 = v16;
        if ( *(_DWORD *)(a1 + 340) <= (unsigned int)v16 )
        {
          v67 = (unsigned __int64 *)sub_C8D7D0(v83, a1 + 344, 0, 0x18u, (unsigned __int64 *)&v113, a6);
          v68 = &v67[3 * *(unsigned int *)(a1 + 336)];
          if ( v68 )
          {
            *v68 = 6;
            v68[1] = 0;
            v68[2] = v19;
            if ( v19 != -8192 && v19 != -4096 )
              sub_BD73F0((__int64)v68);
          }
          sub_F17F80(v83, v67);
          v69 = (int)v113;
          v70 = *(_QWORD *)(a1 + 328);
          if ( a1 + 344 != v70 )
            _libc_free(v70);
          ++*(_DWORD *)(a1 + 336);
          *(_QWORD *)(a1 + 328) = v67;
          *(_DWORD *)(a1 + 340) = v69;
        }
        else
        {
          v18 = (_QWORD *)(*(_QWORD *)(a1 + 328) + 24 * v16);
          if ( v18 )
          {
            *v18 = 6;
            v18[1] = 0;
            v18[2] = v19;
            if ( v19 != -8192 && v19 != -4096 )
              sub_BD73F0((__int64)v18);
            v17 = *(_DWORD *)(a1 + 336);
          }
          *(_DWORD *)(a1 + 336) = v17 + 1;
        }
        goto LABEL_21;
      }
      v14 = *(_QWORD *)*v85;
      if ( v19 != v14 )
        goto LABEL_13;
LABEL_21:
      v88 += 2;
      if ( v84 == v88 )
        goto LABEL_5;
    }
    v8 = *(_QWORD *)(v19 + 40);
    sub_23D0AB0((__int64)&v113, v19, 0, 0, 0);
    if ( *(_BYTE *)v19 == 84 )
    {
      v73 = sub_AA5190(v8);
      if ( v73 )
      {
        v74 = v72;
      }
      else
      {
        v74 = 0;
        v71 = 0;
      }
      v75 = v82;
      LOBYTE(v75) = v71;
      BYTE1(v75) = v74;
      v82 = v75;
      sub_A88F30((__int64)&v113, v8, v73, v75);
    }
    sub_2939E80((__int64)&v105, a1, v86);
    v9 = _mm_loadu_si128(&v106);
    v102 = _mm_loadu_si128(&v105);
    v103 = v9;
    v10.m128i_i64[0] = (__int64)sub_BD5D20(v19);
    LOWORD(v107) = 261;
    v105 = v10;
    v14 = (__int64)sub_293ACB0(
                     (__int64 *)&v113,
                     *v85,
                     (__int64)&v102,
                     v11,
                     v12,
                     v13,
                     (__int64 *)v10.m128i_i64[0],
                     v10.m128i_i64[1],
                     v106.m128i_i32[0],
                     v106.m128i_i32[2],
                     261);
    sub_BD6B90((unsigned __int8 *)v14, (unsigned __int8 *)v19);
    nullsub_61();
    v115[14] = &unk_49DA100;
    nullsub_63();
    v15 = (unsigned __int64)v113;
    if ( v113 != v115 )
      goto LABEL_12;
    goto LABEL_13;
  }
  if ( *(_QWORD *)(a1 + 40) || (result = *(unsigned __int8 *)(a1 + 320), (_BYTE)result) )
  {
    v83 = a1 + 328;
LABEL_5:
    *(_DWORD *)(a1 + 56) = 0;
    sub_293AA10(*(_QWORD **)(a1 + 16));
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = a1 + 8;
    *(_QWORD *)(a1 + 32) = a1 + 8;
    *(_QWORD *)(a1 + 40) = 0;
    *(_BYTE *)(a1 + 320) = 0;
    v115[0] = 0;
    sub_F5C6D0(v83, 0, 0, (__int64)&v113);
    if ( v115[0] )
      ((void (__fastcall *)(unsigned __int64 **, unsigned __int64 **, __int64))v115[0])(&v113, &v113, 3);
    return 1;
  }
  return result;
}
