// Function: sub_37F36E0
// Address: 0x37f36e0
//
__int64 __fastcall sub_37F36E0(int a1, unsigned __int64 *a2, __int64 a3, __int64 *a4)
{
  __int64 *v5; // rax
  __int64 *v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r15
  __int64 v13; // rbx
  unsigned __int64 v14; // rdx
  __int64 v15; // r12
  unsigned int v16; // eax
  __int64 v17; // rbx
  __int64 v18; // r15
  __int64 v19; // r9
  int v20; // r10d
  unsigned int v21; // ecx
  _DWORD *v22; // r13
  _DWORD *v23; // rax
  int v24; // edx
  int *v25; // r13
  __int64 v26; // r14
  int v27; // eax
  __int64 v28; // r9
  __int64 v29; // r12
  unsigned int v30; // esi
  int v31; // edx
  int v32; // edi
  unsigned int v33; // r14d
  int v34; // r10d
  int v35; // esi
  __int64 v36; // r12
  __int64 v37; // rbx
  __int64 v38; // r14
  __int64 v39; // rbx
  int v40; // edx
  unsigned int v41; // ecx
  int *v42; // rsi
  int v43; // edi
  __int64 v44; // rdi
  __int64 i; // rbx
  __int64 v46; // rsi
  __int64 v47; // r8
  __int64 v48; // rdx
  unsigned int v49; // r12d
  unsigned int v50; // r13d
  unsigned __int64 v51; // rcx
  unsigned int v52; // eax
  __int64 v53; // rax
  __int64 v55; // rcx
  __int64 v56; // rdi
  int v57; // eax
  int v58; // r8d
  unsigned int v59; // esi
  int *v60; // rdi
  int v61; // r10d
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v66; // r12
  __int64 v67; // rax
  _BYTE *v68; // r14
  _BYTE *v69; // r12
  __int64 v70; // rbx
  __int64 v71; // r15
  int v72; // esi
  int v73; // r10d
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 v77; // rcx
  _BYTE **v78; // r8
  __int64 v79; // rax
  __int64 (*v80)(); // r10
  __int64 v81; // rcx
  __int64 v82; // rdx
  __int64 *v83; // rbx
  __int64 v84; // rax
  __int64 v85; // rdx
  int v86; // r11d
  _DWORD *v87; // r10
  int v88; // edi
  int v89; // r15d
  __int64 v90; // rdi
  __int64 v91; // [rsp+10h] [rbp-190h]
  __int64 v93; // [rsp+20h] [rbp-180h]
  __int64 v94; // [rsp+28h] [rbp-178h]
  _QWORD *v95; // [rsp+38h] [rbp-168h]
  _QWORD *v96; // [rsp+40h] [rbp-160h]
  __int64 v98; // [rsp+50h] [rbp-150h]
  __int64 v99; // [rsp+58h] [rbp-148h]
  __int64 v100; // [rsp+68h] [rbp-138h]
  __int64 v101; // [rsp+68h] [rbp-138h]
  __int64 v103; // [rsp+78h] [rbp-128h]
  __int64 v104; // [rsp+88h] [rbp-118h] BYREF
  __int64 v105; // [rsp+90h] [rbp-110h] BYREF
  __int64 v106; // [rsp+98h] [rbp-108h] BYREF
  __int64 v107; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v108; // [rsp+A8h] [rbp-F8h]
  __int64 v109; // [rsp+B0h] [rbp-F0h]
  unsigned int v110; // [rsp+B8h] [rbp-E8h]
  _BYTE *v111; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 v112; // [rsp+C8h] [rbp-D8h]
  _BYTE v113[208]; // [rsp+D0h] [rbp-D0h] BYREF

  v96 = (_QWORD *)a2[4];
  v5 = (__int64 *)a2[8];
  v93 = *v5;
  if ( a2 == (unsigned __int64 *)*v5 )
    v93 = v5[1];
  v6 = (__int64 *)a2[14];
  v91 = *v6;
  if ( a2 == (unsigned __int64 *)*v6 )
    v91 = v6[1];
  LOBYTE(v112) = 0;
  v7 = sub_2E7AAE0((__int64)v96, a2[2], (__int64)v111, 0);
  v8 = (__int64)(v96 + 40);
  v99 = v7;
  if ( a1 )
  {
    v83 = (__int64 *)a2[1];
    sub_2E33BD0(v8, v7);
    v84 = *(_QWORD *)v99;
    v85 = *v83;
    *(_QWORD *)(v99 + 8) = v83;
    v85 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v99 = v85 | v84 & 7;
    *(_QWORD *)(v85 + 8) = v99;
    *v83 = v99 | *v83 & 7;
  }
  else
  {
    v9 = v7;
    sub_2E33BD0(v8, v7);
    v10 = *a2;
    *(_QWORD *)(v9 + 8) = a2;
    *(_QWORD *)v9 = v10 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v9 & 7LL;
    *(_QWORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v9;
    *a2 = v9 | *a2 & 7;
  }
  v11 = a3;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v98 = v99 + 48;
  v12 = a2[7];
  v110 = 0;
  v95 = a2 + 6;
  if ( a2 + 6 == (unsigned __int64 *)v12 )
    goto LABEL_42;
  do
  {
    v100 = v11;
    v13 = (__int64)sub_2E7B2C0(v96, v12);
    sub_2E31040((__int64 *)(v99 + 40), v13);
    v14 = *(_QWORD *)(v99 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v13 + 8) = v98;
    *(_QWORD *)v13 = v14 | *(_QWORD *)v13 & 7LL;
    *(_QWORD *)(v14 + 8) = v13;
    *(_QWORD *)(v99 + 48) = v13 | *(_QWORD *)(v99 + 48) & 7LL;
    v15 = *(_QWORD *)(v13 + 32);
    v16 = sub_2E88FE0(v13);
    v17 = *(_QWORD *)(v13 + 32);
    v11 = v100;
    v103 = v15 + 40LL * v16;
    if ( v103 == v17 )
      goto LABEL_39;
    v101 = v12;
    v18 = v11;
    do
    {
      v29 = *(unsigned int *)(v17 + 8);
      if ( (unsigned int)(v29 - 1) <= 0x3FFFFFFE )
        goto LABEL_13;
      if ( v110 )
      {
        v19 = v110 - 1;
        v20 = 1;
        v21 = v19 & (37 * v29);
        v22 = (_DWORD *)(v108 + 8LL * v21);
        v23 = 0;
        v24 = *v22;
        if ( (_DWORD)v29 == *v22 )
        {
LABEL_11:
          v25 = v22 + 1;
          goto LABEL_12;
        }
        while ( v24 != -1 )
        {
          if ( v24 == -2 && !v23 )
            v23 = v22;
          v11 = (unsigned int)(v20 + 1);
          v21 = v19 & (v20 + v21);
          v22 = (_DWORD *)(v108 + 8LL * v21);
          v24 = *v22;
          if ( (_DWORD)v29 == *v22 )
            goto LABEL_11;
          ++v20;
        }
        if ( !v23 )
          v23 = v22;
        ++v107;
        v31 = v109 + 1;
        if ( 4 * ((int)v109 + 1) < 3 * v110 )
        {
          if ( v110 - HIDWORD(v109) - v31 <= v110 >> 3 )
          {
            sub_2FFACA0((__int64)&v107, v110);
            if ( !v110 )
            {
LABEL_143:
              LODWORD(v109) = v109 + 1;
              BUG();
            }
            v19 = 0;
            v33 = (v110 - 1) & (37 * v29);
            v34 = 1;
            v31 = v109 + 1;
            v23 = (_DWORD *)(v108 + 8LL * v33);
            v35 = *v23;
            if ( (_DWORD)v29 != *v23 )
            {
              while ( v35 != -1 )
              {
                if ( !v19 && v35 == -2 )
                  v19 = (__int64)v23;
                v11 = (unsigned int)(v34 + 1);
                v33 = (v110 - 1) & (v34 + v33);
                v23 = (_DWORD *)(v108 + 8LL * v33);
                v35 = *v23;
                if ( (_DWORD)v29 == *v23 )
                  goto LABEL_19;
                ++v34;
              }
              if ( v19 )
                v23 = (_DWORD *)v19;
            }
          }
          goto LABEL_19;
        }
      }
      else
      {
        ++v107;
      }
      sub_2FFACA0((__int64)&v107, 2 * v110);
      if ( !v110 )
        goto LABEL_143;
      v19 = v108;
      v30 = (v110 - 1) & (37 * v29);
      v31 = v109 + 1;
      v23 = (_DWORD *)(v108 + 8LL * v30);
      v32 = *v23;
      if ( (_DWORD)v29 != *v23 )
      {
        v86 = 1;
        v87 = 0;
        while ( v32 != -1 )
        {
          if ( !v87 && v32 == -2 )
            v87 = v23;
          v11 = (unsigned int)(v86 + 1);
          v30 = (v110 - 1) & (v86 + v30);
          v23 = (_DWORD *)(v108 + 8LL * v30);
          v32 = *v23;
          if ( (_DWORD)v29 == *v23 )
            goto LABEL_19;
          ++v86;
        }
        if ( v87 )
          v23 = v87;
      }
LABEL_19:
      LODWORD(v109) = v31;
      if ( *v23 != -1 )
        --HIDWORD(v109);
      *v23 = v29;
      v25 = v23 + 1;
      v23[1] = 0;
LABEL_12:
      v26 = 16 * (v29 & 0x7FFFFFFF);
      v27 = sub_2EC06C0(v18, *(_QWORD *)(*(_QWORD *)(v18 + 56) + v26) & 0xFFFFFFFFFFFFFFF8LL, byte_3F871B3, 0, v11, v19);
      *v25 = v27;
      sub_2EAB0C0(v17, v27);
      if ( a1 == 1 )
      {
        v111 = v113;
        v112 = 0x400000000LL;
        v66 = (int)v29 < 0
            ? *(_QWORD *)(*(_QWORD *)(v18 + 56) + v26 + 8)
            : *(_QWORD *)(*(_QWORD *)(v18 + 304) + 8 * v29);
        if ( v66 )
        {
          if ( (*(_BYTE *)(v66 + 3) & 0x10) != 0 )
          {
            while ( 1 )
            {
              v66 = *(_QWORD *)(v66 + 32);
              if ( !v66 )
                break;
              if ( (*(_BYTE *)(v66 + 3) & 0x10) == 0 )
                goto LABEL_90;
            }
          }
          else
          {
LABEL_90:
            v67 = 0;
LABEL_91:
            if ( a2 != *(unsigned __int64 **)(*(_QWORD *)(v66 + 16) + 24LL) )
            {
              if ( v67 + 1 > (unsigned __int64)HIDWORD(v112) )
              {
                sub_C8D5F0((__int64)&v111, v113, v67 + 1, 8u, v11, v28);
                v67 = (unsigned int)v112;
              }
              *(_QWORD *)&v111[8 * v67] = v66;
              v67 = (unsigned int)(v112 + 1);
              LODWORD(v112) = v112 + 1;
            }
            while ( 1 )
            {
              v66 = *(_QWORD *)(v66 + 32);
              if ( !v66 )
                break;
              if ( (*(_BYTE *)(v66 + 3) & 0x10) == 0 )
                goto LABEL_91;
            }
            v68 = v111;
            v69 = &v111[8 * v67];
            if ( v69 != v111 )
            {
              v94 = v17;
              v70 = v18;
              do
              {
                v71 = *(_QWORD *)v68;
                v68 += 8;
                sub_2EBE590(
                  v70,
                  *v25,
                  *(_QWORD *)(*(_QWORD *)(v70 + 56) + 16LL * (*(_DWORD *)(v71 + 8) & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                  0);
                sub_2EAB0C0(v71, *v25);
              }
              while ( v69 != v68 );
              v18 = v70;
              v68 = v111;
              v17 = v94;
            }
            if ( v68 != v113 )
              _libc_free((unsigned __int64)v68);
          }
        }
      }
LABEL_13:
      v17 += 40;
    }
    while ( v17 != v103 );
    v11 = v18;
    v12 = v101;
LABEL_39:
    if ( !v12 )
      BUG();
    if ( (*(_BYTE *)v12 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v12 + 44) & 8) != 0 )
        v12 = *(_QWORD *)(v12 + 8);
    }
    v12 = *(_QWORD *)(v12 + 8);
  }
  while ( v95 != (_QWORD *)v12 );
LABEL_42:
  v36 = sub_2E311E0(v99);
  if ( v36 != v98 )
  {
    while ( 1 )
    {
      v37 = *(_QWORD *)(v36 + 32);
      v38 = v37 + 40LL * (*(_DWORD *)(v36 + 40) & 0xFFFFFF);
      v39 = v37 + 40LL * (unsigned int)sub_2E88FE0(v36);
      if ( v38 != v39 )
        break;
LABEL_51:
      if ( (*(_BYTE *)v36 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v36 + 44) & 8) != 0 )
          v36 = *(_QWORD *)(v36 + 8);
      }
      v36 = *(_QWORD *)(v36 + 8);
      if ( v98 == v36 )
        goto LABEL_53;
    }
    while ( 1 )
    {
LABEL_46:
      if ( *(_BYTE *)v39 )
        goto LABEL_45;
      v40 = *(_DWORD *)(v39 + 8);
      if ( !v110 )
        goto LABEL_45;
      v41 = (v110 - 1) & (37 * v40);
      v42 = (int *)(v108 + 8LL * v41);
      v43 = *v42;
      if ( v40 != *v42 )
        break;
LABEL_49:
      if ( v42 == (int *)(v108 + 8LL * v110) )
        goto LABEL_45;
      v44 = v39;
      v39 += 40;
      sub_2EAB0C0(v44, v42[1]);
      if ( v38 == v39 )
        goto LABEL_51;
    }
    v72 = 1;
    while ( v43 != -1 )
    {
      v73 = v72 + 1;
      v41 = (v110 - 1) & (v72 + v41);
      v42 = (int *)(v108 + 8LL * v41);
      v43 = *v42;
      if ( v40 == *v42 )
        goto LABEL_49;
      v72 = v73;
    }
LABEL_45:
    v39 += 40;
    if ( v38 == v39 )
      goto LABEL_51;
    goto LABEL_46;
  }
LABEL_53:
  for ( i = *(_QWORD *)(v99 + 56); *(_WORD *)(i + 68) == 68 || !*(_WORD *)(i + 68); i = *(_QWORD *)(i + 8) )
  {
    v46 = *(_QWORD *)(i + 32);
    v47 = 120;
    if ( v93 != *(_QWORD *)(v46 + 104) )
      v47 = 40;
    v48 = *(_QWORD *)(*(_QWORD *)(i + 24) + 56LL);
    v49 = 2 * (v93 != *(_QWORD *)(v46 + 104)) + 1;
    v50 = 2 * (v93 == *(_QWORD *)(v46 + 104)) + 1;
    v51 = a2[7];
    if ( v48 != i )
    {
      v52 = 0;
      do
      {
        v48 = *(_QWORD *)(v48 + 8);
        ++v52;
      }
      while ( v48 != i );
      if ( v52 )
      {
        v53 = v52 - 1LL;
        do
          v51 = *(_QWORD *)(v51 + 8);
        while ( v53-- != 0 );
      }
    }
    v55 = *(_QWORD *)(v51 + 32);
    v56 = v46 + v47;
    if ( a1 )
    {
      sub_2EAB0C0(v56, *(_DWORD *)(v55 + v47 + 8));
      sub_2E8A650(i, v49 + 1);
      sub_2E8A650(i, v49);
    }
    else
    {
      v57 = *(_DWORD *)(v56 + 8);
      v58 = v57;
      if ( v110 )
      {
        v59 = (v110 - 1) & (37 * v57);
        v60 = (int *)(v108 + 8LL * v59);
        v61 = *v60;
        if ( v57 == *v60 )
        {
LABEL_66:
          if ( v60 != (int *)(v108 + 8LL * v110) )
            v58 = v60[1];
        }
        else
        {
          v88 = 1;
          while ( v61 != -1 )
          {
            v89 = v88 + 1;
            v90 = (v110 - 1) & (v59 + v88);
            v59 = v90;
            v60 = (int *)(v108 + 8 * v90);
            v61 = *v60;
            if ( v57 == *v60 )
              goto LABEL_66;
            v88 = v89;
          }
        }
      }
      sub_2EAB0C0(v55 + 40LL * v49, v58);
      sub_2E8A650(i, v50 + 1);
      sub_2E8A650(i, v50);
    }
    if ( (*(_BYTE *)i & 4) == 0 )
    {
      while ( (*(_BYTE *)(i + 44) & 8) != 0 )
        i = *(_QWORD *)(i + 8);
    }
  }
  v104 = 0;
  if ( a1 )
  {
    sub_2E33690((__int64)a2, v91, v99);
    sub_2E32770(v91, (__int64)a2, v99);
    sub_2E33F80(v99, v91, -1, v74, v75, v76);
    v105 = 0;
    v112 = 0x400000000LL;
    v78 = &v111;
    v111 = v113;
    v106 = 0;
    v79 = *a4;
    v80 = *(__int64 (**)())(*a4 + 344);
    if ( v80 != sub_2DB1AE0 )
    {
      ((void (__fastcall *)(__int64 *, unsigned __int64 *, __int64 *, __int64 *, _BYTE **, _QWORD))v80)(
        a4,
        a2,
        &v105,
        &v106,
        &v111,
        0);
      v79 = *a4;
    }
    (*(void (__fastcall **)(__int64 *, unsigned __int64 *, _QWORD, __int64, _BYTE **))(v79 + 360))(a4, a2, 0, v77, v78);
    v81 = v106;
    v82 = v105;
    if ( v106 == v91 )
      v81 = v99;
    if ( v105 == v91 )
      v82 = v99;
    (*(void (__fastcall **)(__int64 *, unsigned __int64 *, __int64, __int64, _BYTE *, _QWORD, __int64 *, _QWORD))(*a4 + 368))(
      a4,
      a2,
      v82,
      v81,
      v111,
      (unsigned int)v112,
      &v104,
      0);
    if ( (*(unsigned int (__fastcall **)(__int64 *, __int64, _QWORD))(*a4 + 360))(a4, v99, 0) )
      (*(void (__fastcall **)(__int64 *, __int64, __int64, _QWORD, _QWORD, _QWORD, __int64 *, _QWORD))(*a4 + 368))(
        a4,
        v99,
        v91,
        0,
        0,
        0,
        &v104,
        0);
    if ( v111 != v113 )
      _libc_free((unsigned __int64)v111);
  }
  else
  {
    sub_2E337A0(v93, (__int64)a2, v99);
    sub_2E33F80(v99, (__int64)a2, -1, v62, v63, v64);
    sub_2E32770((__int64)a2, v93, v99);
    sub_2E32A60(v93, (__int64)a2);
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a4 + 360))(a4, v99, 0);
    (*(void (__fastcall **)(__int64 *, __int64, unsigned __int64 *, _QWORD, _QWORD, _QWORD, __int64 *, _QWORD))(*a4 + 368))(
      a4,
      v99,
      a2,
      0,
      0,
      0,
      &v104,
      0);
  }
  if ( v104 )
    sub_B91220((__int64)&v104, v104);
  sub_C7D6A0(v108, 8LL * v110, 4);
  return v99;
}
