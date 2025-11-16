// Function: sub_1EB5340
// Address: 0x1eb5340
//
__int64 __fastcall sub_1EB5340(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // rcx
  int v9; // esi
  __int64 v10; // rdi
  __int64 v11; // rdx
  unsigned int v12; // r14d
  unsigned __int16 *v13; // rax
  unsigned __int16 *v14; // r8
  __int64 v15; // rdx
  int v16; // eax
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // rax
  unsigned int *v20; // r15
  unsigned int *v21; // r11
  char *v22; // r14
  unsigned __int64 *v23; // r8
  __int64 v24; // r10
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r15
  __int64 v28; // r13
  unsigned int v29; // esi
  __int16 v30; // ax
  _WORD *v31; // rsi
  unsigned __int16 *v32; // rdx
  __int64 v33; // r12
  __int64 v34; // r15
  int v35; // eax
  char *v36; // rsi
  __int64 v37; // r14
  __int64 v38; // rbx
  __int64 v39; // r12
  float v40; // xmm1_4
  __int64 v41; // r15
  float v42; // xmm0_4
  __int64 v43; // rax
  __int64 v44; // r12
  unsigned __int64 v45; // rdi
  __int64 v46; // r13
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rdx
  __int64 v50; // rsi
  __int64 v51; // rdi
  __int64 v52; // r9
  __int64 v53; // rcx
  __int64 (*v54)(void); // rsi
  __int64 v55; // rax
  int v56; // eax
  unsigned int *v58; // r15
  int v59; // eax
  __int64 v60; // rax
  __int64 v61; // rcx
  __int64 v62; // rsi
  __int64 v63; // rdx
  __int64 (*v64)(void); // rcx
  __int64 v65; // rax
  int v66; // eax
  unsigned int *v67; // [rsp+8h] [rbp-228h]
  unsigned int *v68; // [rsp+20h] [rbp-210h]
  unsigned __int64 *v69; // [rsp+20h] [rbp-210h]
  __int64 v70; // [rsp+28h] [rbp-208h]
  __int64 v71; // [rsp+28h] [rbp-208h]
  __int64 v72; // [rsp+28h] [rbp-208h]
  unsigned __int16 v73; // [rsp+30h] [rbp-200h]
  __int64 v74; // [rsp+30h] [rbp-200h]
  unsigned __int16 *v75; // [rsp+38h] [rbp-1F8h]
  unsigned int *v77; // [rsp+50h] [rbp-1E0h] BYREF
  __int64 v78; // [rsp+58h] [rbp-1D8h]
  _BYTE v79[32]; // [rsp+60h] [rbp-1D0h] BYREF
  unsigned __int16 *v80; // [rsp+80h] [rbp-1B0h] BYREF
  unsigned int v81; // [rsp+88h] [rbp-1A8h]
  char v82; // [rsp+90h] [rbp-1A0h] BYREF
  __int64 v83; // [rsp+B0h] [rbp-180h]
  int v84; // [rsp+B8h] [rbp-178h]
  int v85; // [rsp+C0h] [rbp-170h]
  char v86; // [rsp+C4h] [rbp-16Ch]
  char *v87; // [rsp+D0h] [rbp-160h] BYREF
  __int64 v88; // [rsp+D8h] [rbp-158h]
  char v89; // [rsp+E0h] [rbp-150h] BYREF
  void *v90; // [rsp+120h] [rbp-110h] BYREF
  __int64 v91; // [rsp+128h] [rbp-108h]
  __int64 v92; // [rsp+130h] [rbp-100h]
  __int64 v93; // [rsp+138h] [rbp-F8h]
  __int64 v94; // [rsp+140h] [rbp-F0h]
  __int64 v95; // [rsp+148h] [rbp-E8h]
  __int64 v96; // [rsp+150h] [rbp-E0h]
  __int64 v97; // [rsp+158h] [rbp-D8h]
  int v98; // [rsp+160h] [rbp-D0h]
  char v99; // [rsp+164h] [rbp-CCh]
  __int64 v100; // [rsp+168h] [rbp-C8h]
  __int64 v101; // [rsp+170h] [rbp-C0h]
  _BYTE *v102; // [rsp+178h] [rbp-B8h]
  _BYTE *v103; // [rsp+180h] [rbp-B0h]
  __int64 v104; // [rsp+188h] [rbp-A8h]
  int v105; // [rsp+190h] [rbp-A0h]
  _BYTE v106[32]; // [rsp+198h] [rbp-98h] BYREF
  __int64 v107; // [rsp+1B8h] [rbp-78h]
  _BYTE *v108; // [rsp+1C0h] [rbp-70h]
  _BYTE *v109; // [rsp+1C8h] [rbp-68h]
  __int64 v110; // [rsp+1D0h] [rbp-60h]
  int v111; // [rsp+1D8h] [rbp-58h]
  _BYTE v112[80]; // [rsp+1E0h] [rbp-50h] BYREF

  v3 = a2;
  v4 = a1;
  v5 = *(unsigned int *)(a2 + 112);
  v6 = *(_QWORD *)(a1 + 256);
  v7 = *(_QWORD *)(a1 + 272);
  v77 = (unsigned int *)v79;
  v78 = 0x800000000LL;
  sub_20C72B0(&v80, v5, v6, a1 + 280, v7);
LABEL_2:
  v9 = v85;
  if ( v85 < 0 )
  {
    while ( 1 )
    {
      v85 = v9 + 1;
      v12 = v80[v81 + (__int64)v9];
LABEL_14:
      if ( !v12 )
        break;
      v16 = sub_21038C0(*(_QWORD *)(v4 + 272), v3, v12, v8);
      if ( !v16 )
        goto LABEL_65;
      if ( v16 != 1 )
        goto LABEL_2;
      v19 = (unsigned int)v78;
      if ( (unsigned int)v78 >= HIDWORD(v78) )
      {
        sub_16CD150((__int64)&v77, v79, 0, 4, v17, v18);
        v19 = (unsigned int)v78;
      }
      v77[v19] = v12;
      v9 = v85;
      LODWORD(v78) = v78 + 1;
      if ( v85 >= 0 )
        goto LABEL_3;
    }
  }
  else
  {
LABEL_3:
    if ( !v86 )
    {
      v10 = 2LL * v9;
      while ( 1 )
      {
        while ( 1 )
        {
LABEL_5:
          if ( v84 <= v9 )
            goto LABEL_38;
          v85 = ++v9;
          v11 = 2LL * v81;
          v12 = *(unsigned __int16 *)(v83 + v10);
          v13 = v80;
          v14 = &v80[(unsigned __int64)v11 / 2];
          v8 = v11 >> 1;
          v15 = v11 >> 3;
          if ( !v15 )
            break;
          v8 = (__int64)&v80[4 * v15];
          while ( v12 != *v13 )
          {
            if ( v12 == v13[1] )
            {
              v10 += 2;
              if ( v14 == v13 + 1 )
                goto LABEL_14;
              goto LABEL_5;
            }
            if ( v12 == v13[2] )
            {
              v10 += 2;
              if ( v14 == v13 + 2 )
                goto LABEL_14;
              goto LABEL_5;
            }
            if ( v12 == v13[3] )
            {
              v10 += 2;
              if ( v14 == v13 + 3 )
                goto LABEL_14;
              goto LABEL_5;
            }
            v13 += 4;
            if ( (unsigned __int16 *)v8 == v13 )
            {
              v8 = v14 - v13;
              goto LABEL_22;
            }
          }
LABEL_13:
          v10 += 2;
          if ( v14 == v13 )
            goto LABEL_14;
        }
LABEL_22:
        if ( v8 != 2 )
        {
          if ( v8 != 3 )
          {
            if ( v8 != 1 )
              goto LABEL_14;
            goto LABEL_29;
          }
          if ( v12 == *v13 )
            goto LABEL_13;
          ++v13;
        }
        if ( v12 == *v13 )
          goto LABEL_13;
        ++v13;
LABEL_29:
        if ( v12 == *v13 )
        {
          v10 += 2;
          if ( v14 != v13 )
            continue;
        }
        goto LABEL_14;
      }
    }
  }
LABEL_38:
  v20 = v77;
  v21 = &v77[(unsigned int)v78];
  if ( v21 == v77 )
  {
    v40 = unk_4530D80;
LABEL_87:
    if ( v40 == *(float *)(v3 + 116) )
    {
      v12 = -1;
    }
    else
    {
      v60 = *(_QWORD *)(v4 + 680);
      v91 = v3;
      v61 = *(_QWORD *)(v4 + 256);
      v62 = *(_QWORD *)(v4 + 264);
      v90 = &unk_4A00C10;
      v92 = a3;
      v63 = *(_QWORD *)(v60 + 40);
      v94 = v62;
      v93 = v63;
      v95 = v61;
      v64 = *(__int64 (**)(void))(**(_QWORD **)(v60 + 16) + 40LL);
      v65 = 0;
      if ( v64 != sub_1D00B00 )
      {
        v65 = v64();
        v63 = v93;
      }
      v96 = v65;
      v97 = v4 + 672;
      v66 = *(_DWORD *)(a3 + 8);
      v99 = 0;
      v100 = v4 + 376;
      v98 = v66;
      v102 = v106;
      v103 = v106;
      v101 = 0;
      v104 = 4;
      v105 = 0;
      v107 = 0;
      v108 = v112;
      v109 = v112;
      v110 = 4;
      v111 = 0;
      *(_QWORD *)(v63 + 8) = &v90;
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v4 + 688) + 24LL))(*(_QWORD *)(v4 + 688));
      *(_QWORD *)(v93 + 8) = 0;
      if ( v109 != v108 )
        _libc_free((unsigned __int64)v109);
      if ( v103 != v102 )
        _libc_free((unsigned __int64)v103);
      v12 = 0;
    }
  }
  else
  {
    v22 = &v89;
    v23 = (unsigned __int64 *)&v87;
    v24 = v4;
LABEL_40:
    v25 = *(_QWORD *)(v24 + 240);
    v26 = *v20;
    v87 = v22;
    v88 = 0x800000000LL;
    if ( !v25 )
      BUG();
    v67 = v21;
    v70 = (__int64)v23;
    v68 = v20;
    v27 = v3;
    v28 = v24;
    v29 = *(_DWORD *)(*(_QWORD *)(v25 + 8) + 24LL * (unsigned int)v26 + 16);
    v30 = v26 * (v29 & 0xF);
    v31 = (_WORD *)(*(_QWORD *)(v25 + 56) + 2LL * (v29 >> 4));
    v32 = v31 + 1;
    v73 = *v31 + v30;
LABEL_42:
    v75 = v32;
    v33 = v27;
    while ( v75 )
    {
      v34 = sub_2103840(*(_QWORD *)(v28 + 272), v33, v73, v26, v23, 0x800000000LL);
      sub_20FD0B0(v34, 0xFFFFFFFFLL);
      v35 = *(_DWORD *)(v34 + 120);
      if ( v35 )
      {
        v36 = v22;
        v37 = v28;
        v38 = 8LL * (unsigned int)(v35 - 1);
        v3 = v33;
        v39 = v34;
        v40 = unk_4530D80;
        while ( 1 )
        {
          v41 = *(_QWORD *)(*(_QWORD *)(v39 + 112) + v38);
          v42 = *(float *)(v41 + 116);
          if ( v42 == INFINITY || v42 > *(float *)(v3 + 116) )
            break;
          v43 = (unsigned int)v88;
          if ( (unsigned int)v88 >= HIDWORD(v88) )
          {
            sub_16CD150(v70, v36, 0, 8, (int)v23, 0);
            v43 = (unsigned int)v88;
          }
          v38 -= 8;
          *(_QWORD *)&v87[8 * v43] = v41;
          LODWORD(v88) = v88 + 1;
          if ( v38 == -8 )
          {
            v33 = v3;
            v28 = v37;
            v22 = v36;
            goto LABEL_76;
          }
        }
        v24 = v37;
        v21 = v67;
        v22 = v36;
        v58 = v68;
        v23 = (unsigned __int64 *)v70;
        if ( v87 != v36 )
        {
          v69 = (unsigned __int64 *)v70;
          v72 = v24;
          _libc_free((unsigned __int64)v87);
          v21 = v67;
          v24 = v72;
          v23 = v69;
        }
        v20 = v58 + 1;
        if ( v20 == v21 )
        {
          v4 = v24;
          goto LABEL_87;
        }
        goto LABEL_40;
      }
LABEL_76:
      v32 = 0;
      v59 = *v75++;
      v26 = v59 + (unsigned int)v73;
      if ( !(_WORD)v59 )
      {
        v27 = v33;
        goto LABEL_42;
      }
      v73 += v59;
    }
    v44 = v28;
    v45 = (unsigned __int64)v87;
    if ( (_DWORD)v88 )
    {
      v46 = v28 + 376;
      v74 = 8LL * (unsigned int)v88;
      do
      {
        v47 = *(_QWORD *)((char *)v75 + v45);
        v48 = *(_DWORD *)(v47 + 112) & 0x7FFFFFFF;
        v49 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v44 + 256) + 264LL) + 4 * v48);
        if ( (_DWORD)v49 )
        {
          v71 = *(_QWORD *)((char *)v75 + v45);
          sub_21031A0(*(_QWORD *)(v44 + 272), v47, v49, v48, v23);
          v50 = *(_QWORD *)(v44 + 680);
          v51 = *(_QWORD *)(v44 + 256);
          v91 = v71;
          v90 = &unk_4A00C10;
          v52 = *(_QWORD *)(v44 + 264);
          v92 = a3;
          v53 = *(_QWORD *)(v50 + 40);
          v94 = v52;
          v93 = v53;
          v95 = v51;
          v54 = *(__int64 (**)(void))(**(_QWORD **)(v50 + 16) + 40LL);
          v55 = 0;
          if ( v54 != sub_1D00B00 )
          {
            v55 = v54();
            v53 = v93;
          }
          v96 = v55;
          v97 = v44 + 672;
          v56 = *(_DWORD *)(a3 + 8);
          v99 = 0;
          v100 = v46;
          v98 = v56;
          v102 = v106;
          v103 = v106;
          v101 = 0;
          v104 = 4;
          v105 = 0;
          v107 = 0;
          v108 = v112;
          v109 = v112;
          v110 = 4;
          v111 = 0;
          *(_QWORD *)(v53 + 8) = &v90;
          (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v44 + 688) + 24LL))(*(_QWORD *)(v44 + 688));
          *(_QWORD *)(v93 + 8) = 0;
          if ( v109 != v108 )
            _libc_free((unsigned __int64)v109);
          if ( v103 != v102 )
            _libc_free((unsigned __int64)v103);
          v45 = (unsigned __int64)v87;
        }
        v75 += 4;
      }
      while ( (unsigned __int16 *)v74 != v75 );
    }
    if ( (char *)v45 != v22 )
      _libc_free(v45);
    v12 = *v68;
  }
LABEL_65:
  if ( v80 != (unsigned __int16 *)&v82 )
    _libc_free((unsigned __int64)v80);
  if ( v77 != (unsigned int *)v79 )
    _libc_free((unsigned __int64)v77);
  return v12;
}
