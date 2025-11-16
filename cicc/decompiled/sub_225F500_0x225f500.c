// Function: sub_225F500
// Address: 0x225f500
//
__int64 __fastcall sub_225F500(__int64 a1, int a2, int a3, const char **a4, __m128i a5)
{
  int v7; // edx
  __int64 v8; // r15
  int v9; // r14d
  __int64 v10; // r15
  int v11; // r14d
  int v12; // r15d
  __int64 v13; // r15
  int v14; // r14d
  char *v15; // rsi
  char *v16; // rax
  unsigned int v17; // r12d
  char *v18; // r13
  unsigned __int64 v19; // r8
  __int64 v20; // r15
  __int64 v21; // rbx
  _QWORD *v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rcx
  unsigned __int64 v25; // r8
  __int64 v26; // r13
  __int64 v27; // rbx
  _QWORD *v28; // rdi
  unsigned int v30; // eax
  unsigned __int64 v31; // rdx
  unsigned int v32; // eax
  char *v33; // r12
  size_t v34; // rax
  __int64 v35; // [rsp+8h] [rbp-318h]
  int v36; // [rsp+40h] [rbp-2E0h] BYREF
  int v37; // [rsp+44h] [rbp-2DCh] BYREF
  int v38; // [rsp+48h] [rbp-2D8h] BYREF
  int v39; // [rsp+4Ch] [rbp-2D4h] BYREF
  unsigned int v40; // [rsp+50h] [rbp-2D0h] BYREF
  int v41; // [rsp+54h] [rbp-2CCh] BYREF
  char *s; // [rsp+58h] [rbp-2C8h] BYREF
  __int64 v43; // [rsp+60h] [rbp-2C0h] BYREF
  __int64 v44; // [rsp+68h] [rbp-2B8h] BYREF
  __int64 v45; // [rsp+70h] [rbp-2B0h] BYREF
  __int64 v46; // [rsp+78h] [rbp-2A8h] BYREF
  char *v47; // [rsp+80h] [rbp-2A0h] BYREF
  __int64 v48; // [rsp+88h] [rbp-298h] BYREF
  __int64 v49; // [rsp+90h] [rbp-290h] BYREF
  __int64 v50; // [rsp+98h] [rbp-288h] BYREF
  __int64 v51; // [rsp+A0h] [rbp-280h] BYREF
  __int64 v52; // [rsp+A8h] [rbp-278h] BYREF
  __int64 v53; // [rsp+B0h] [rbp-270h] BYREF
  __int64 v54; // [rsp+B8h] [rbp-268h] BYREF
  _BYTE *v55[2]; // [rsp+C0h] [rbp-260h] BYREF
  _QWORD v56[2]; // [rsp+D0h] [rbp-250h] BYREF
  _DWORD v57[4]; // [rsp+E0h] [rbp-240h] BYREF
  _QWORD *v58; // [rsp+F0h] [rbp-230h]
  __int64 v59; // [rsp+F8h] [rbp-228h]
  _BYTE v60[16]; // [rsp+100h] [rbp-220h] BYREF
  _QWORD *v61; // [rsp+110h] [rbp-210h]
  __int64 v62; // [rsp+118h] [rbp-208h]
  _BYTE v63[16]; // [rsp+120h] [rbp-200h] BYREF
  _QWORD *v64; // [rsp+130h] [rbp-1F0h]
  __int64 v65; // [rsp+138h] [rbp-1E8h]
  _BYTE v66[16]; // [rsp+140h] [rbp-1E0h] BYREF
  _QWORD *v67; // [rsp+150h] [rbp-1D0h]
  __int64 v68; // [rsp+158h] [rbp-1C8h]
  _BYTE v69[16]; // [rsp+160h] [rbp-1C0h] BYREF
  _QWORD *v70; // [rsp+170h] [rbp-1B0h]
  __int64 v71; // [rsp+178h] [rbp-1A8h]
  _BYTE v72[16]; // [rsp+180h] [rbp-1A0h] BYREF
  _QWORD *v73; // [rsp+190h] [rbp-190h]
  __int64 v74; // [rsp+198h] [rbp-188h]
  _BYTE v75[16]; // [rsp+1A0h] [rbp-180h] BYREF
  unsigned __int64 v76; // [rsp+1B0h] [rbp-170h]
  __int64 v77; // [rsp+1B8h] [rbp-168h]
  __int64 v78; // [rsp+1C0h] [rbp-160h]
  void *v79; // [rsp+1D0h] [rbp-150h] BYREF
  _BYTE v80[16]; // [rsp+1D8h] [rbp-148h] BYREF
  __int64 *v81; // [rsp+1E8h] [rbp-138h]
  __int64 v82; // [rsp+1F8h] [rbp-128h] BYREF
  __int64 *v83; // [rsp+208h] [rbp-118h]
  __int64 v84; // [rsp+218h] [rbp-108h] BYREF
  __int64 *v85; // [rsp+228h] [rbp-F8h]
  __int64 v86; // [rsp+238h] [rbp-E8h] BYREF
  __int64 *v87; // [rsp+248h] [rbp-D8h]
  __int64 v88; // [rsp+258h] [rbp-C8h] BYREF
  __int64 *v89; // [rsp+268h] [rbp-B8h]
  __int64 v90; // [rsp+278h] [rbp-A8h] BYREF
  __int64 *v91; // [rsp+288h] [rbp-98h]
  __int64 v92; // [rsp+298h] [rbp-88h] BYREF
  unsigned __int64 v93; // [rsp+2A8h] [rbp-78h]
  unsigned int v94; // [rsp+2B0h] [rbp-70h]
  int v95; // [rsp+2B4h] [rbp-6Ch]
  int *v96; // [rsp+2C8h] [rbp-58h]
  __int64 v97; // [rsp+2D0h] [rbp-50h] BYREF
  bool v98[8]; // [rsp+2D8h] [rbp-48h] BYREF
  __int64 v99; // [rsp+2E0h] [rbp-40h]
  __int64 v100; // [rsp+2E8h] [rbp-38h]

  v58 = v60;
  v61 = v63;
  v64 = v66;
  v67 = v69;
  s = 0;
  v40 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v47 = 0;
  v57[2] = 0;
  v59 = 0;
  v60[0] = 0;
  v62 = 0;
  v63[0] = 0;
  v65 = 0;
  v66[0] = 0;
  v68 = 0;
  v69[0] = 0;
  v7 = *(_DWORD *)(a1 + 176);
  v70 = v72;
  v73 = v75;
  v78 = 0x1000000000LL;
  v71 = 0;
  v72[0] = 0;
  v74 = 0;
  v75[0] = 0;
  v76 = 0;
  v77 = 0;
  if ( !(unsigned int)sub_967C50(
                        a3,
                        a4,
                        v7,
                        &v36,
                        &v43,
                        &v37,
                        &v44,
                        &v38,
                        &v45,
                        &v39,
                        &v46,
                        (int *)&v40,
                        (__int64 *)&v47,
                        v57) )
  {
    v57[0] = a2;
    v8 = v43;
    v9 = v36;
    if ( v36 != (_DWORD)v49 || v43 != v50 )
    {
      sub_95D500(&v49, &v50);
      LODWORD(v49) = v9;
      v50 = v8;
    }
    v10 = v44;
    v11 = v37;
    if ( v37 != (_DWORD)v51 || v44 != v52 )
    {
      sub_95D500(&v51, &v52);
      LODWORD(v51) = v11;
      v52 = v10;
    }
    v12 = v38;
    if ( v38 != v11 || v45 != v52 )
    {
      v35 = v45;
      sub_95D500(&v51, &v52);
      LODWORD(v51) = v12;
      v52 = v35;
    }
    v13 = v46;
    v14 = v39;
    if ( v39 != (_DWORD)v53 || v46 != v54 )
    {
      sub_95D500(&v53, &v54);
      LODWORD(v53) = v14;
      v54 = v13;
    }
    sub_B6EEA0(&v48);
    v40 |= 1u;
    v79 = &unk_4A08338;
    sub_2258B00((__int64)v80, (__int64)v57);
    v96 = &v36;
    v99 = 0;
    v97 = v43;
    v98[0] = v57[0] == 0;
    v100 = 0;
    v79 = &unk_4A31DF8;
    if ( v36 <= 0 )
      goto LABEL_15;
    v55[0] = &v41;
    *(_QWORD *)(__readfsqword(0) - 24) = v55;
    *(_QWORD *)(__readfsqword(0) - 32) = sub_2257A90;
    if ( &_pthread_key_create )
    {
      v32 = pthread_once(&dword_4FD6B64, init_routine);
      if ( !v32 )
      {
        nullsub_1744(&v79);
        if ( v100 )
          (*(void (__fastcall **)(__int64, int *, __int64 *, _QWORD))(*(_QWORD *)v100 + 16LL))(v100, v96, &v97, 0);
        sub_2C83470(v98, (unsigned int)*v96, v97, byte_3F871B3);
LABEL_15:
        v15 = (char *)&v41;
        v16 = sub_225A270((__int64 *)a1, &v41, v40, &v48, (__int64)v57, a5);
        v17 = v41;
        v18 = v16;
        if ( v41 )
        {
          if ( v41 == 9 )
          {
            v41 = 6;
            v17 = 6;
          }
        }
        else
        {
          v55[0] = v56;
          sub_2257AB0((__int64 *)v55, *((_BYTE **)v16 + 95), *((_QWORD *)v16 + 95) + *((_QWORD *)v16 + 96));
          if ( (unsigned __int8)sub_2C74B60(v55, 1, 0) )
            sub_BA9520((__int64)v18, v55[0], (__int64)v55[1]);
          v30 = sub_309EEA0(v18, &s);
          v15 = s;
          v41 = v30;
          v17 = v30;
          if ( s )
          {
            v15 = s;
            v31 = strlen(s);
            if ( v31 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 88) )
              sub_4262D8((__int64)"basic_string::append");
            sub_2241490((unsigned __int64 *)(a1 + 80), v15, v31);
            if ( s )
              j_j___libc_free_0_0((unsigned __int64)s);
            s = 0;
            v17 = v41;
          }
          if ( (_QWORD *)v55[0] != v56 )
          {
            v15 = (char *)(v56[0] + 1LL);
            j_j___libc_free_0((unsigned __int64)v55[0]);
          }
        }
        v79 = &unk_4A08338;
        if ( v100 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v100 + 8LL))(v100);
        sub_2C835D0(v98);
        if ( v95 )
        {
          v19 = v93;
          if ( v94 )
          {
            v20 = 8LL * v94;
            v21 = 0;
            do
            {
              v22 = *(_QWORD **)(v19 + v21);
              if ( v22 != (_QWORD *)-8LL && v22 )
              {
                v15 = (char *)(*v22 + 17LL);
                sub_C7D6A0((__int64)v22, (__int64)v15, 8);
                v19 = v93;
              }
              v21 += 8;
            }
            while ( v20 != v21 );
          }
        }
        else
        {
          v19 = v93;
        }
        _libc_free(v19);
        if ( v91 != &v92 )
        {
          v15 = (char *)(v92 + 1);
          j_j___libc_free_0((unsigned __int64)v91);
        }
        if ( v89 != &v90 )
        {
          v15 = (char *)(v90 + 1);
          j_j___libc_free_0((unsigned __int64)v89);
        }
        if ( v87 != &v88 )
        {
          v15 = (char *)(v88 + 1);
          j_j___libc_free_0((unsigned __int64)v87);
        }
        if ( v85 != &v86 )
        {
          v15 = (char *)(v86 + 1);
          j_j___libc_free_0((unsigned __int64)v85);
        }
        if ( v83 != &v84 )
        {
          v15 = (char *)(v84 + 1);
          j_j___libc_free_0((unsigned __int64)v83);
        }
        if ( v81 != &v82 )
        {
          v15 = (char *)(v82 + 1);
          j_j___libc_free_0((unsigned __int64)v81);
        }
        if ( v18 )
        {
          sub_BA9C10((_QWORD **)v18, (__int64)v15, v23, v24);
          j_j___libc_free_0((unsigned __int64)v18);
        }
        sub_B6E710(&v48);
        goto LABEL_42;
      }
    }
    else
    {
      v32 = -1;
    }
    sub_4264C5(v32);
  }
  v33 = v47;
  if ( v47 )
  {
    v34 = strlen(v47);
    sub_2241130((unsigned __int64 *)(a1 + 80), 0, *(_QWORD *)(a1 + 88), v33, v34);
    if ( v47 )
      j_j___libc_free_0_0((unsigned __int64)v47);
  }
  v17 = 7;
LABEL_42:
  v25 = v76;
  if ( HIDWORD(v77) && (_DWORD)v77 )
  {
    v26 = 8LL * (unsigned int)v77;
    v27 = 0;
    do
    {
      v28 = *(_QWORD **)(v25 + v27);
      if ( v28 != (_QWORD *)-8LL && v28 )
      {
        sub_C7D6A0((__int64)v28, *v28 + 17LL, 8);
        v25 = v76;
      }
      v27 += 8;
    }
    while ( v26 != v27 );
  }
  _libc_free(v25);
  if ( v73 != (_QWORD *)v75 )
    j_j___libc_free_0((unsigned __int64)v73);
  if ( v70 != (_QWORD *)v72 )
    j_j___libc_free_0((unsigned __int64)v70);
  if ( v67 != (_QWORD *)v69 )
    j_j___libc_free_0((unsigned __int64)v67);
  if ( v64 != (_QWORD *)v66 )
    j_j___libc_free_0((unsigned __int64)v64);
  if ( v61 != (_QWORD *)v63 )
    j_j___libc_free_0((unsigned __int64)v61);
  if ( v58 != (_QWORD *)v60 )
    j_j___libc_free_0((unsigned __int64)v58);
  sub_95D500(&v53, &v54);
  sub_95D500(&v51, &v52);
  sub_95D500(&v49, &v50);
  return v17;
}
