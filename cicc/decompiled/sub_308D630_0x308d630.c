// Function: sub_308D630
// Address: 0x308d630
//
char __fastcall sub_308D630(
        _QWORD *a1,
        unsigned __int32 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        __int64 a9,
        unsigned int a10)
{
  int v13; // eax
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rax
  int v17; // eax
  __int64 v18; // rdx
  unsigned __int8 *v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rsi
  __int64 v24; // r11
  __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rcx
  _QWORD *v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // r12
  int v34; // eax
  unsigned __int8 *v35; // rsi
  __int64 v36; // rax
  unsigned __int32 v37; // r8d
  __int64 v38; // rsi
  __int64 v39; // rax
  unsigned int v40; // eax
  __int64 v41; // r10
  __int32 v42; // r8d
  unsigned int v43; // esi
  __int64 v44; // r8
  int v45; // r11d
  _QWORD *v46; // rdx
  unsigned int v47; // edi
  __int64 v48; // rcx
  _DWORD *v49; // rdx
  __int64 v50; // rdi
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rcx
  _QWORD *v54; // rax
  __int64 v55; // rdx
  bool v56; // r9
  unsigned int v57; // esi
  __int64 v58; // r8
  int v59; // r11d
  unsigned int v60; // edi
  __int64 v61; // rcx
  int v62; // eax
  int v63; // ecx
  bool v64; // al
  bool v65; // r8
  int v66; // eax
  int v67; // eax
  int v68; // eax
  __int64 v69; // rdi
  int v70; // r9d
  unsigned int v71; // r12d
  _QWORD *v72; // r8
  __int64 v73; // rsi
  int v74; // eax
  int v75; // esi
  __int64 v76; // r8
  unsigned int v77; // eax
  __int64 v78; // rdi
  int v79; // r11d
  _QWORD *v80; // r9
  bool v81; // al
  int v82; // eax
  int v83; // esi
  __int64 v84; // r8
  unsigned int v85; // eax
  __int64 v86; // rdi
  int v87; // r11d
  int v88; // eax
  int v89; // eax
  __int64 v90; // rdi
  int v91; // r9d
  unsigned int v92; // r12d
  __int64 v93; // rsi
  __int64 v95; // [rsp+0h] [rbp-C0h]
  __int64 v96; // [rsp+0h] [rbp-C0h]
  __int64 v97; // [rsp+8h] [rbp-B8h]
  __int64 v98; // [rsp+8h] [rbp-B8h]
  __int64 v99; // [rsp+8h] [rbp-B8h]
  int v100; // [rsp+10h] [rbp-B0h]
  unsigned __int32 v101; // [rsp+10h] [rbp-B0h]
  unsigned __int32 v102; // [rsp+10h] [rbp-B0h]
  __int64 v103; // [rsp+10h] [rbp-B0h]
  unsigned __int32 v104; // [rsp+18h] [rbp-A8h]
  int v105; // [rsp+18h] [rbp-A8h]
  __int64 v106; // [rsp+18h] [rbp-A8h]
  unsigned __int8 *v107; // [rsp+28h] [rbp-98h] BYREF
  unsigned __int8 *v108; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int8 *v109; // [rsp+38h] [rbp-88h] BYREF
  unsigned __int8 *v110; // [rsp+40h] [rbp-80h] BYREF
  __int64 v111; // [rsp+48h] [rbp-78h]
  __int64 v112; // [rsp+50h] [rbp-70h]
  __m128i v113; // [rsp+60h] [rbp-60h] BYREF
  __int64 v114; // [rsp+70h] [rbp-50h]
  __int64 v115; // [rsp+78h] [rbp-48h]
  __int64 v116; // [rsp+80h] [rbp-40h]

  if ( (unsigned int)*(unsigned __int16 *)(a9 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(a9 + 32) + 64LL) & 8) != 0
    || ((v13 = *(_DWORD *)(a9 + 44), (v13 & 4) != 0) || (v13 & 8) == 0
      ? (v14 = (*(_QWORD *)(*(_QWORD *)(a9 + 16) + 24LL) >> 19) & 1LL)
      : (LOBYTE(v14) = sub_2E88A90(a9, 0x80000, 1)),
        (_BYTE)v14) )
  {
    v34 = sub_3089AC0((__int64)a1, a9);
    v35 = *(unsigned __int8 **)(a9 + 56);
    v105 = v34;
    v36 = *(_QWORD *)(a9 + 32);
    v107 = v35;
    v37 = *(_DWORD *)(v36 + 40LL * a10 + 8);
    if ( v35 )
    {
      v101 = *(_DWORD *)(v36 + 40LL * a10 + 8);
      sub_B96E90((__int64)&v107, (__int64)v35, 1);
      v36 = *(_QWORD *)(a9 + 32);
      v37 = v101;
    }
    v102 = v37;
    v38 = a1[57];
    v98 = *(_QWORD *)(v36 + 40LL * (unsigned int)(v105 + 5) + 24);
    v39 = *(unsigned int *)(*(_QWORD *)(v38 + 312)
                          + 16LL
                          * (*(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[56] + 56LL)
                                                                         + 16LL * (v37 & 0x7FFFFFFF))
                                                             & 0xFFFFFFFFFFFFFFF8LL)
                                                 + 24LL)
                           + *(_DWORD *)(v38 + 328)
                           * (unsigned int)((__int64)(*(_QWORD *)(v38 + 288) - *(_QWORD *)(v38 + 280)) >> 3)));
    v113.m128i_i8[8] = 0;
    v113.m128i_i64[0] = v39;
    v40 = sub_CA1930(&v113);
    v41 = *(_QWORD *)(a9 + 24);
    v42 = v102;
    if ( (unsigned int)v98 < v40 )
    {
      v50 = *(_QWORD *)(*(_QWORD *)(a9 + 32) + 40LL * (unsigned int)(v105 + 4) + 24);
      if ( v40 == 16 && (_DWORD)v98 == 8 )
      {
        v51 = -45280;
        if ( (_DWORD)v50 != 1 )
          v51 = -47760;
      }
      else
      {
        v56 = v40 == 32;
        if ( (_DWORD)v98 == 8 && v40 == 32 )
        {
          v51 = -45760;
          if ( (_DWORD)v50 != 1 )
            v51 = -48240;
        }
        else
        {
          v64 = v40 == 64;
          if ( (_DWORD)v98 == 8 && v64 )
          {
            v51 = -46240;
            if ( (_DWORD)v50 != 1 )
              v51 = -48720;
          }
          else if ( v56 && (_DWORD)v98 == 16 )
          {
            v51 = -45640;
            if ( (_DWORD)v50 != 1 )
              v51 = -48120;
          }
          else if ( v64 && (_DWORD)v98 == 16 )
          {
            v51 = -46120;
            if ( (_DWORD)v50 != 1 )
              v51 = -48600;
          }
          else
          {
            if ( (_DWORD)v98 != 32 || !v64 )
              goto LABEL_176;
            v51 = -46160;
            if ( (_DWORD)v50 != 1 )
              v51 = -48640;
          }
        }
      }
      v52 = *(_QWORD *)(a1[58] + 8LL) + v51;
      v109 = v107;
      v53 = v52;
      if ( v107 )
      {
        v99 = v41;
        v96 = v52;
        sub_B96E90((__int64)&v109, (__int64)v107, 1);
        v42 = v102;
        v41 = v99;
        v110 = v109;
        v53 = v96;
        if ( v109 )
        {
          sub_B976B0((__int64)&v109, v109, (__int64)&v110);
          v53 = v96;
          v109 = 0;
          v41 = v99;
          v42 = v102;
        }
      }
      else
      {
        v110 = 0;
      }
      v111 = 0;
      v112 = 0;
      v54 = sub_2F2A600(v41, a9, (__int64 *)&v110, v53, v42);
      v113.m128i_i64[0] = 0;
      v114 = 0;
      v103 = (__int64)v54;
      v106 = v55;
      v113.m128i_i32[2] = a2;
      v115 = 0;
      v116 = 0;
      sub_2E8EAD0(v55, (__int64)v54, &v113);
      v113.m128i_i64[0] = 1;
      v114 = 0;
      v115 = 0;
      sub_2E8EAD0(v106, v103, &v113);
      if ( v110 )
        sub_B91220((__int64)&v110, (__int64)v110);
      if ( v109 )
        sub_B91220((__int64)&v109, (__int64)v109);
    }
    else
    {
      (*(void (__fastcall **)(_QWORD, _QWORD, __int64, unsigned __int8 **, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(*(_QWORD *)a1[58] + 496LL))(
        a1[58],
        *(_QWORD *)(a9 + 24),
        a9,
        &v107,
        v102,
        a2,
        0,
        0,
        0);
    }
    if ( v107 )
      sub_B91220((__int64)&v107, (__int64)v107);
    v43 = *(_DWORD *)(a3 + 24);
    if ( v43 )
    {
      v44 = *(_QWORD *)(a3 + 8);
      v45 = 1;
      v46 = 0;
      v47 = (v43 - 1) & (((unsigned int)a9 >> 9) ^ ((unsigned int)a9 >> 4));
      v16 = v44 + 16LL * v47;
      v48 = *(_QWORD *)v16;
      if ( a9 == *(_QWORD *)v16 )
        goto LABEL_34;
      while ( v48 != -4096 )
      {
        if ( !v46 && v48 == -8192 )
          v46 = (_QWORD *)v16;
        v47 = (v43 - 1) & (v45 + v47);
        v16 = v44 + 16LL * v47;
        v48 = *(_QWORD *)v16;
        if ( a9 == *(_QWORD *)v16 )
          goto LABEL_34;
        ++v45;
      }
      if ( !v46 )
        v46 = (_QWORD *)v16;
      v66 = *(_DWORD *)(a3 + 16);
      ++*(_QWORD *)a3;
      v63 = v66 + 1;
      if ( 4 * (v66 + 1) < 3 * v43 )
      {
        if ( v43 - *(_DWORD *)(a3 + 20) - v63 > v43 >> 3 )
          goto LABEL_68;
        sub_2E261E0(a3, v43);
        v67 = *(_DWORD *)(a3 + 24);
        if ( v67 )
        {
          v68 = v67 - 1;
          v69 = *(_QWORD *)(a3 + 8);
          v70 = 1;
          v71 = v68 & (((unsigned int)a9 >> 9) ^ ((unsigned int)a9 >> 4));
          v72 = 0;
          v63 = *(_DWORD *)(a3 + 16) + 1;
          v46 = (_QWORD *)(v69 + 16LL * v71);
          v73 = *v46;
          if ( a9 == *v46 )
            goto LABEL_68;
          while ( v73 != -4096 )
          {
            if ( v73 == -8192 && !v72 )
              v72 = v46;
            v71 = v68 & (v70 + v71);
            v46 = (_QWORD *)(v69 + 16LL * v71);
            v73 = *v46;
            if ( a9 == *v46 )
              goto LABEL_68;
            ++v70;
          }
LABEL_99:
          if ( v72 )
            v46 = v72;
          goto LABEL_68;
        }
        goto LABEL_175;
      }
    }
    else
    {
      ++*(_QWORD *)a3;
    }
    sub_2E261E0(a3, 2 * v43);
    v74 = *(_DWORD *)(a3 + 24);
    if ( v74 )
    {
      v75 = v74 - 1;
      v76 = *(_QWORD *)(a3 + 8);
      v77 = (v74 - 1) & (((unsigned int)a9 >> 9) ^ ((unsigned int)a9 >> 4));
      v63 = *(_DWORD *)(a3 + 16) + 1;
      v46 = (_QWORD *)(v76 + 16LL * v77);
      v78 = *v46;
      if ( a9 == *v46 )
        goto LABEL_68;
      v79 = 1;
      v80 = 0;
      while ( v78 != -4096 )
      {
        if ( v78 == -8192 && !v80 )
          v80 = v46;
        v77 = v75 & (v79 + v77);
        v46 = (_QWORD *)(v76 + 16LL * v77);
        v78 = *v46;
        if ( a9 == *v46 )
          goto LABEL_68;
        ++v79;
      }
      goto LABEL_109;
    }
LABEL_175:
    ++*(_DWORD *)(a3 + 16);
    BUG();
  }
  if ( (unsigned int)*(unsigned __int16 *)(a9 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(a9 + 32) + 64LL) & 0x10) != 0
    || ((v15 = *(_DWORD *)(a9 + 44), (v15 & 4) == 0) && (v15 & 8) != 0
      ? (LOBYTE(v16) = sub_2E88A90(a9, 0x100000, 1))
      : (v16 = (*(_QWORD *)(*(_QWORD *)(a9 + 16) + 24LL) >> 20) & 1LL),
        (_BYTE)v16) )
  {
    v17 = sub_3089AC0((__int64)a1, a9);
    v18 = *(_QWORD *)(a9 + 32);
    v19 = *(unsigned __int8 **)(a9 + 56);
    v100 = v17;
    v108 = v19;
    v104 = *(_DWORD *)(v18 + 40LL * a10 + 8);
    if ( v19 )
      sub_B96E90((__int64)&v108, (__int64)v19, 1);
    v20 = a1[57];
    v21 = *(unsigned int *)(*(_QWORD *)(v20 + 312)
                          + 16LL
                          * (*(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[56] + 56LL)
                                                                         + 16LL * (v104 & 0x7FFFFFFF))
                                                             & 0xFFFFFFFFFFFFFFF8LL)
                                                 + 24LL)
                           + *(_DWORD *)(v20 + 328)
                           * (unsigned int)((__int64)(*(_QWORD *)(v20 + 288) - *(_QWORD *)(v20 + 280)) >> 3)));
    v113.m128i_i8[8] = 0;
    v113.m128i_i64[0] = v21;
    v22 = sub_CA1930(&v113);
    v23 = *(_QWORD *)(a9 + 32);
    v24 = *(_QWORD *)(a9 + 24);
    v25 = *(_QWORD *)(v23 + 40LL * (unsigned int)(v100 + 5) + 24);
    if ( v22 <= (unsigned int)v25 )
    {
      (*(void (__fastcall **)(_QWORD, _QWORD, __int64, unsigned __int8 **, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(*(_QWORD *)a1[58] + 496LL))(
        a1[58],
        *(_QWORD *)(a9 + 24),
        a9,
        &v108,
        a2,
        v104,
        0,
        0,
        0);
LABEL_59:
      if ( v108 )
        sub_B91220((__int64)&v108, (__int64)v108);
      v57 = *(_DWORD *)(a3 + 24);
      if ( !v57 )
      {
        ++*(_QWORD *)a3;
LABEL_128:
        sub_2E261E0(a3, 2 * v57);
        v82 = *(_DWORD *)(a3 + 24);
        if ( !v82 )
          goto LABEL_177;
        v83 = v82 - 1;
        v84 = *(_QWORD *)(a3 + 8);
        v85 = (v82 - 1) & (((unsigned int)a9 >> 9) ^ ((unsigned int)a9 >> 4));
        v63 = *(_DWORD *)(a3 + 16) + 1;
        v46 = (_QWORD *)(v84 + 16LL * v85);
        v86 = *v46;
        if ( a9 == *v46 )
          goto LABEL_68;
        v87 = 1;
        v80 = 0;
        while ( v86 != -4096 )
        {
          if ( !v80 && v86 == -8192 )
            v80 = v46;
          v85 = v83 & (v87 + v85);
          v46 = (_QWORD *)(v84 + 16LL * v85);
          v86 = *v46;
          if ( a9 == *v46 )
            goto LABEL_68;
          ++v87;
        }
LABEL_109:
        if ( v80 )
          v46 = v80;
        goto LABEL_68;
      }
      v58 = *(_QWORD *)(a3 + 8);
      v59 = 1;
      v46 = 0;
      v60 = (v57 - 1) & (((unsigned int)a9 >> 9) ^ ((unsigned int)a9 >> 4));
      v16 = v58 + 16LL * v60;
      v61 = *(_QWORD *)v16;
      if ( a9 != *(_QWORD *)v16 )
      {
        while ( v61 != -4096 )
        {
          if ( !v46 && v61 == -8192 )
            v46 = (_QWORD *)v16;
          v60 = (v57 - 1) & (v59 + v60);
          v16 = v58 + 16LL * v60;
          v61 = *(_QWORD *)v16;
          if ( a9 == *(_QWORD *)v16 )
            goto LABEL_34;
          ++v59;
        }
        if ( !v46 )
          v46 = (_QWORD *)v16;
        v62 = *(_DWORD *)(a3 + 16);
        ++*(_QWORD *)a3;
        v63 = v62 + 1;
        if ( 4 * (v62 + 1) < 3 * v57 )
        {
          if ( v57 - *(_DWORD *)(a3 + 20) - v63 > v57 >> 3 )
          {
LABEL_68:
            *(_DWORD *)(a3 + 16) = v63;
            if ( *v46 != -4096 )
              --*(_DWORD *)(a3 + 20);
            *v46 = a9;
            LODWORD(v16) = 1;
            v49 = v46 + 1;
            *v49 = 0;
            goto LABEL_35;
          }
          sub_2E261E0(a3, v57);
          v88 = *(_DWORD *)(a3 + 24);
          if ( v88 )
          {
            v89 = v88 - 1;
            v90 = *(_QWORD *)(a3 + 8);
            v91 = 1;
            v92 = v89 & (((unsigned int)a9 >> 9) ^ ((unsigned int)a9 >> 4));
            v72 = 0;
            v63 = *(_DWORD *)(a3 + 16) + 1;
            v46 = (_QWORD *)(v90 + 16LL * v92);
            v93 = *v46;
            if ( a9 == *v46 )
              goto LABEL_68;
            while ( v93 != -4096 )
            {
              if ( !v72 && v93 == -8192 )
                v72 = v46;
              v92 = v89 & (v91 + v92);
              v46 = (_QWORD *)(v90 + 16LL * v92);
              v93 = *v46;
              if ( a9 == *v46 )
                goto LABEL_68;
              ++v91;
            }
            goto LABEL_99;
          }
LABEL_177:
          ++*(_DWORD *)(a3 + 16);
          BUG();
        }
        goto LABEL_128;
      }
LABEL_34:
      v49 = (_DWORD *)(v16 + 8);
      LODWORD(v16) = *(_DWORD *)(v16 + 8) + 1;
LABEL_35:
      *v49 = v16;
      return v16;
    }
    v26 = *(_QWORD *)(v23 + 40LL * (unsigned int)(v100 + 4) + 24);
    if ( v22 == 16 && (_DWORD)v25 == 8 )
    {
      v27 = -46600;
      if ( (_DWORD)v26 != 1 )
        v27 = -49080;
LABEL_19:
      v28 = *(_QWORD *)(a1[58] + 8LL) + v27;
      v109 = v108;
      v29 = v28;
      if ( v108 )
      {
        v97 = v24;
        v95 = v28;
        sub_B96E90((__int64)&v109, (__int64)v108, 1);
        v24 = v97;
        v29 = v95;
        v110 = v109;
        if ( v109 )
        {
          sub_B976B0((__int64)&v109, v109, (__int64)&v110);
          v29 = v95;
          v109 = 0;
          v24 = v97;
        }
      }
      else
      {
        v110 = 0;
      }
      v111 = 0;
      v112 = 0;
      v30 = sub_2F2A600(v24, a9, (__int64 *)&v110, v29, a2);
      v113.m128i_i64[0] = 0;
      v31 = (__int64)v30;
      v33 = v32;
      v113.m128i_i32[2] = v104;
      v114 = 0;
      v115 = 0;
      v116 = 0;
      sub_2E8EAD0(v32, (__int64)v30, &v113);
      v113.m128i_i64[0] = 1;
      v114 = 0;
      v115 = 0;
      sub_2E8EAD0(v33, v31, &v113);
      if ( v110 )
        sub_B91220((__int64)&v110, (__int64)v110);
      if ( v109 )
        sub_B91220((__int64)&v109, (__int64)v109);
      goto LABEL_59;
    }
    v65 = v22 == 32;
    if ( (_DWORD)v25 == 8 && v22 == 32 )
    {
      v27 = -46640;
      if ( (_DWORD)v26 != 1 )
        v27 = -49120;
      goto LABEL_19;
    }
    v81 = v22 == 64;
    if ( (_DWORD)v25 == 8 && v81 )
    {
      v27 = -46680;
      if ( (_DWORD)v26 != 1 )
        v27 = -49160;
      goto LABEL_19;
    }
    if ( v65 && (_DWORD)v25 == 16 )
    {
      v27 = -45200;
      if ( (_DWORD)v26 != 1 )
        v27 = -47680;
      goto LABEL_19;
    }
    if ( v81 && (_DWORD)v25 == 16 )
    {
      v27 = -45240;
      if ( (_DWORD)v26 != 1 )
        v27 = -47720;
      goto LABEL_19;
    }
    if ( (_DWORD)v25 == 32 && v81 )
    {
      v27 = -45720;
      if ( (_DWORD)v26 != 1 )
        v27 = -48200;
      goto LABEL_19;
    }
LABEL_176:
    BUG();
  }
  return v16;
}
