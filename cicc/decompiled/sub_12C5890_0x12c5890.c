// Function: sub_12C5890
// Address: 0x12c5890
//
__int64 __fastcall sub_12C5890(__int64 a1, int a2, int a3, __int64 a4)
{
  __int64 p_s; // rsi
  int v8; // edx
  __int64 v9; // r15
  int v10; // r14d
  __int64 v11; // r15
  int v12; // r14d
  int v13; // r15d
  __int64 v14; // r15
  int v15; // r14d
  __int64 v16; // rax
  unsigned int v17; // r12d
  __int64 v18; // r13
  __int64 v19; // r8
  __int64 v20; // r15
  __int64 v21; // rbx
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r8
  __int64 v28; // r13
  __int64 v29; // rbx
  __int64 v30; // rdi
  unsigned int v32; // eax
  char *v33; // r15
  size_t v34; // rdx
  __int64 v35; // rcx
  unsigned int v36; // eax
  char *v37; // r12
  size_t v38; // rax
  __int64 v39; // [rsp+8h] [rbp-328h]
  char v40; // [rsp+4Bh] [rbp-2E5h] BYREF
  int v41; // [rsp+4Ch] [rbp-2E4h] BYREF
  int v42; // [rsp+50h] [rbp-2E0h] BYREF
  int v43; // [rsp+54h] [rbp-2DCh] BYREF
  int v44; // [rsp+58h] [rbp-2D8h] BYREF
  unsigned int v45; // [rsp+5Ch] [rbp-2D4h] BYREF
  char *s; // [rsp+60h] [rbp-2D0h] BYREF
  __int64 v47; // [rsp+68h] [rbp-2C8h] BYREF
  __int64 v48; // [rsp+70h] [rbp-2C0h] BYREF
  __int64 v49; // [rsp+78h] [rbp-2B8h] BYREF
  __int64 v50; // [rsp+80h] [rbp-2B0h] BYREF
  char *v51; // [rsp+88h] [rbp-2A8h] BYREF
  _BYTE v52[8]; // [rsp+90h] [rbp-2A0h] BYREF
  char *v53; // [rsp+98h] [rbp-298h] BYREF
  __int64 v54; // [rsp+A0h] [rbp-290h] BYREF
  __int64 v55; // [rsp+A8h] [rbp-288h] BYREF
  __int64 v56; // [rsp+B0h] [rbp-280h] BYREF
  __int64 v57; // [rsp+B8h] [rbp-278h] BYREF
  __int64 v58; // [rsp+C0h] [rbp-270h] BYREF
  __int64 v59; // [rsp+C8h] [rbp-268h] BYREF
  _DWORD v60[4]; // [rsp+D0h] [rbp-260h] BYREF
  _QWORD *v61; // [rsp+E0h] [rbp-250h]
  __int64 v62; // [rsp+E8h] [rbp-248h]
  _QWORD v63[2]; // [rsp+F0h] [rbp-240h] BYREF
  _QWORD *v64; // [rsp+100h] [rbp-230h]
  __int64 v65; // [rsp+108h] [rbp-228h]
  _QWORD v66[2]; // [rsp+110h] [rbp-220h] BYREF
  _QWORD *v67; // [rsp+120h] [rbp-210h]
  __int64 v68; // [rsp+128h] [rbp-208h]
  _QWORD v69[2]; // [rsp+130h] [rbp-200h] BYREF
  _QWORD *v70; // [rsp+140h] [rbp-1F0h]
  __int64 v71; // [rsp+148h] [rbp-1E8h]
  _QWORD v72[2]; // [rsp+150h] [rbp-1E0h] BYREF
  _QWORD *v73; // [rsp+160h] [rbp-1D0h]
  __int64 v74; // [rsp+168h] [rbp-1C8h]
  _QWORD v75[2]; // [rsp+170h] [rbp-1C0h] BYREF
  _QWORD *v76; // [rsp+180h] [rbp-1B0h]
  __int64 v77; // [rsp+188h] [rbp-1A8h]
  _QWORD v78[2]; // [rsp+190h] [rbp-1A0h] BYREF
  __int64 v79; // [rsp+1A0h] [rbp-190h]
  __int64 v80; // [rsp+1A8h] [rbp-188h]
  __int64 v81; // [rsp+1B0h] [rbp-180h]
  void *v82; // [rsp+1D0h] [rbp-160h] BYREF
  _BYTE v83[16]; // [rsp+1D8h] [rbp-158h] BYREF
  __int64 *v84; // [rsp+1E8h] [rbp-148h]
  __int64 v85; // [rsp+1F8h] [rbp-138h] BYREF
  __int64 *v86; // [rsp+208h] [rbp-128h]
  __int64 v87; // [rsp+218h] [rbp-118h] BYREF
  __int64 *v88; // [rsp+228h] [rbp-108h]
  __int64 v89; // [rsp+238h] [rbp-F8h] BYREF
  __int64 *v90; // [rsp+248h] [rbp-E8h]
  __int64 v91; // [rsp+258h] [rbp-D8h] BYREF
  __int64 *v92; // [rsp+268h] [rbp-C8h]
  __int64 v93; // [rsp+278h] [rbp-B8h] BYREF
  __int64 *v94; // [rsp+288h] [rbp-A8h]
  __int64 v95; // [rsp+298h] [rbp-98h] BYREF
  __int64 v96; // [rsp+2A8h] [rbp-88h]
  unsigned int v97; // [rsp+2B0h] [rbp-80h]
  int v98; // [rsp+2B4h] [rbp-7Ch]
  int *v99; // [rsp+2D0h] [rbp-60h]
  __int64 v100; // [rsp+2D8h] [rbp-58h] BYREF
  bool v101[8]; // [rsp+2E0h] [rbp-50h] BYREF
  __int64 v102; // [rsp+2E8h] [rbp-48h]
  __int64 v103; // [rsp+2F0h] [rbp-40h]

  p_s = a4;
  v61 = v63;
  v64 = v66;
  v67 = v69;
  v70 = v72;
  s = 0;
  v45 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v51 = 0;
  v60[2] = 0;
  v62 = 0;
  LOBYTE(v63[0]) = 0;
  v65 = 0;
  LOBYTE(v66[0]) = 0;
  v68 = 0;
  LOBYTE(v69[0]) = 0;
  v71 = 0;
  LOBYTE(v72[0]) = 0;
  v8 = *(_DWORD *)(a1 + 176);
  v73 = v75;
  v76 = v78;
  v81 = 0x1000000000LL;
  v74 = 0;
  LOBYTE(v75[0]) = 0;
  v77 = 0;
  LOBYTE(v78[0]) = 0;
  v79 = 0;
  v80 = 0;
  if ( !(unsigned int)sub_12D2AA0(
                        a3,
                        a4,
                        v8,
                        (unsigned int)&v41,
                        (unsigned int)&v47,
                        (unsigned int)&v42,
                        (__int64)&v48,
                        (__int64)&v43,
                        (__int64)&v49,
                        (__int64)&v44,
                        (__int64)&v50,
                        (__int64)&v45,
                        (__int64)&v51,
                        (__int64)v60) )
  {
    v60[0] = a2;
    v9 = v47;
    v10 = v41;
    if ( v41 != (_DWORD)v54 || v47 != v55 )
    {
      sub_12C7AC0(&v54, &v55);
      LODWORD(v54) = v10;
      v55 = v9;
    }
    v11 = v48;
    v12 = v42;
    if ( v42 != (_DWORD)v56 || v48 != v57 )
    {
      sub_12C7AC0(&v56, &v57);
      LODWORD(v56) = v12;
      v57 = v11;
    }
    v13 = v43;
    if ( v43 != v12 || v49 != v57 )
    {
      v39 = v49;
      sub_12C7AC0(&v56, &v57);
      LODWORD(v56) = v13;
      v57 = v39;
    }
    v14 = v50;
    v15 = v44;
    if ( v44 != (_DWORD)v58 || v50 != v59 )
    {
      sub_12C7AC0(&v58, &v59);
      LODWORD(v58) = v15;
      v59 = v14;
    }
    sub_1602D10(v52);
    v45 |= 1u;
    v82 = &unk_49E6A40;
    sub_12BE2E0((__int64)v83, (__int64)v60);
    v99 = &v41;
    v102 = 0;
    v100 = v47;
    v101[0] = v60[0] == 0;
    v103 = 0;
    v82 = &unk_49E7FF0;
    if ( v41 <= 0 )
      goto LABEL_15;
    v53 = &v40;
    *(_QWORD *)(__readfsqword(0) - 24) = &v53;
    *(_QWORD *)(__readfsqword(0) - 32) = sub_12BCC20;
    if ( &_pthread_key_create )
    {
      v36 = pthread_once(&dword_4F92D9C, init_routine);
      if ( !v36 )
      {
        nullsub_501(&v82);
        if ( v103 )
          (*(void (__fastcall **)(__int64, int *, __int64 *, _QWORD))(*(_QWORD *)v103 + 16LL))(v103, v99, &v100, 0);
        sub_1C427B0(v101, (unsigned int)*v99, v100, byte_3F871B3);
LABEL_15:
        p_s = (__int64)&v53;
        v16 = sub_12C06E0((_QWORD *)a1, &v53, v45, (__int64)v52, (__int64)v60);
        v17 = (unsigned int)v53;
        v18 = v16;
        if ( (_DWORD)v53 )
        {
          if ( (_DWORD)v53 == 9 )
          {
            LODWORD(v53) = 6;
            v17 = 6;
          }
        }
        else
        {
          p_s = (__int64)&s;
          v32 = sub_12F9390(v16, &s);
          v33 = s;
          LODWORD(v53) = v32;
          v17 = v32;
          if ( s )
          {
            v34 = strlen(s);
            if ( v34 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 88) )
              sub_4262D8((__int64)"basic_string::append");
            p_s = (__int64)v33;
            sub_2241490(a1 + 80, v33, v34, v35);
            if ( s )
              j_j___libc_free_0_0(s);
            s = 0;
            v17 = (unsigned int)v53;
          }
        }
        v82 = &unk_49E6A40;
        if ( v103 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v103 + 8LL))(v103);
        sub_1C428D0(v101);
        if ( v98 )
        {
          v19 = v96;
          if ( v97 )
          {
            v20 = 8LL * v97;
            v21 = 0;
            do
            {
              v22 = *(_QWORD *)(v19 + v21);
              if ( v22 != -8 && v22 )
              {
                _libc_free(v22, p_s);
                v19 = v96;
              }
              v21 += 8;
            }
            while ( v20 != v21 );
          }
        }
        else
        {
          v19 = v96;
        }
        _libc_free(v19, p_s);
        if ( v94 != &v95 )
        {
          p_s = v95 + 1;
          j_j___libc_free_0(v94, v95 + 1);
        }
        if ( v92 != &v93 )
        {
          p_s = v93 + 1;
          j_j___libc_free_0(v92, v93 + 1);
        }
        if ( v90 != &v91 )
        {
          p_s = v91 + 1;
          j_j___libc_free_0(v90, v91 + 1);
        }
        if ( v88 != &v89 )
        {
          p_s = v89 + 1;
          j_j___libc_free_0(v88, v89 + 1);
        }
        if ( v86 != &v87 )
        {
          p_s = v87 + 1;
          j_j___libc_free_0(v86, v87 + 1);
        }
        if ( v84 != &v85 )
        {
          p_s = v85 + 1;
          j_j___libc_free_0(v84, v85 + 1);
        }
        if ( v18 )
        {
          sub_1633490(v18);
          p_s = 736;
          j_j___libc_free_0(v18, 736);
        }
        sub_16025D0(v52, p_s, v23, v24, v25, v26);
        goto LABEL_42;
      }
    }
    else
    {
      v36 = -1;
    }
    sub_4264C5(v36);
  }
  v37 = v51;
  if ( v51 )
  {
    v38 = strlen(v51);
    p_s = 0;
    sub_2241130(a1 + 80, 0, *(_QWORD *)(a1 + 88), v37, v38);
    if ( v51 )
      j_j___libc_free_0_0(v51);
  }
  v17 = 7;
LABEL_42:
  v27 = v79;
  if ( HIDWORD(v80) && (_DWORD)v80 )
  {
    v28 = 8LL * (unsigned int)v80;
    v29 = 0;
    do
    {
      v30 = *(_QWORD *)(v27 + v29);
      if ( v30 != -8 && v30 )
      {
        _libc_free(v30, p_s);
        v27 = v79;
      }
      v29 += 8;
    }
    while ( v28 != v29 );
  }
  _libc_free(v27, p_s);
  if ( v76 != v78 )
    j_j___libc_free_0(v76, v78[0] + 1LL);
  if ( v73 != v75 )
    j_j___libc_free_0(v73, v75[0] + 1LL);
  if ( v70 != v72 )
    j_j___libc_free_0(v70, v72[0] + 1LL);
  if ( v67 != v69 )
    j_j___libc_free_0(v67, v69[0] + 1LL);
  if ( v64 != v66 )
    j_j___libc_free_0(v64, v66[0] + 1LL);
  if ( v61 != v63 )
    j_j___libc_free_0(v61, v63[0] + 1LL);
  sub_12C7AC0(&v58, &v59);
  sub_12C7AC0(&v56, &v57);
  sub_12C7AC0(&v54, &v55);
  return v17;
}
