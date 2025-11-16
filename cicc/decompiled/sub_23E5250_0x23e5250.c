// Function: sub_23E5250
// Address: 0x23e5250
//
void __fastcall sub_23E5250(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        unsigned __int8 a7,
        char a8,
        unsigned int a9,
        __int64 a10)
{
  __int64 v12; // r15
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r15
  __int64 v17; // rbx
  __int64 v18; // rdx
  _BYTE *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned int *v22; // r12
  unsigned int *v23; // r15
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // r15
  _BYTE *v27; // r10
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 *v32; // rdi
  _QWORD *v33; // rax
  __int64 v34; // r15
  unsigned int *v35; // rax
  int v36; // ecx
  unsigned int *v37; // rdx
  __int64 v38; // rdx
  int v39; // r14d
  unsigned int *v40; // r15
  unsigned int *v41; // r14
  __int64 v42; // rdx
  unsigned int v43; // esi
  __int64 v44; // rbx
  unsigned int *v45; // r12
  unsigned int *v46; // r15
  __int64 v47; // rdx
  unsigned int v48; // esi
  unsigned __int64 v49; // rsi
  __int64 v50; // [rsp-10h] [rbp-1C0h]
  __int64 v51; // [rsp-8h] [rbp-1B8h]
  __int64 v52; // [rsp+0h] [rbp-1B0h]
  __int64 v55; // [rsp+18h] [rbp-198h]
  _BYTE *v56; // [rsp+18h] [rbp-198h]
  __int64 v57; // [rsp+18h] [rbp-198h]
  __int64 v58; // [rsp+30h] [rbp-180h]
  __int64 v59; // [rsp+30h] [rbp-180h]
  __int64 v60; // [rsp+30h] [rbp-180h]
  __int64 v62; // [rsp+48h] [rbp-168h]
  _BYTE v65[32]; // [rsp+60h] [rbp-150h] BYREF
  __int16 v66; // [rsp+80h] [rbp-130h]
  _BYTE *v67; // [rsp+90h] [rbp-120h] BYREF
  __int64 v68; // [rsp+98h] [rbp-118h]
  __int64 v69; // [rsp+A0h] [rbp-110h]
  __int16 v70; // [rsp+B0h] [rbp-100h]
  _QWORD v71[4]; // [rsp+C0h] [rbp-F0h] BYREF
  __int16 v72; // [rsp+E0h] [rbp-D0h]
  unsigned int *v73; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v74; // [rsp+F8h] [rbp-B8h]
  _BYTE v75[32]; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v76; // [rsp+120h] [rbp-90h]
  __int64 v77; // [rsp+128h] [rbp-88h]
  __int64 v78; // [rsp+130h] [rbp-80h]
  _QWORD *v79; // [rsp+138h] [rbp-78h]
  void **v80; // [rsp+140h] [rbp-70h]
  void **v81; // [rsp+148h] [rbp-68h]
  __int64 v82; // [rsp+150h] [rbp-60h]
  int v83; // [rsp+158h] [rbp-58h]
  __int16 v84; // [rsp+15Ch] [rbp-54h]
  char v85; // [rsp+15Eh] [rbp-52h]
  __int64 v86; // [rsp+160h] [rbp-50h]
  __int64 v87; // [rsp+168h] [rbp-48h]
  void *v88; // [rsp+170h] [rbp-40h] BYREF
  void *v89; // [rsp+178h] [rbp-38h] BYREF

  v79 = (_QWORD *)sub_BD5C60(a3);
  v80 = &v88;
  v81 = &v89;
  v73 = (unsigned int *)v75;
  v88 = &unk_49DA100;
  v74 = 0x200000000LL;
  v84 = 512;
  LOWORD(v78) = 0;
  v89 = &unk_49DA0B0;
  v82 = 0;
  v83 = 0;
  v85 = 7;
  v86 = 0;
  v87 = 0;
  v76 = 0;
  v77 = 0;
  sub_D5F1F0((__int64)&v73, a3);
  v12 = sub_B43CB0(a3);
  sub_B33910(v71, (__int64 *)&v73);
  v13 = v71[0];
  if ( v71[0] )
    goto LABEL_2;
  v31 = sub_B92180(v12);
  if ( !v31 )
    goto LABEL_3;
  v32 = (__int64 *)(*(_QWORD *)(v31 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(v31 + 8) & 4) != 0 )
    v32 = (__int64 *)*v32;
  v33 = sub_B01860(v32, 0, 0, v31, 0, 0, 0, 1);
  sub_B10CB0(v71, (__int64)v33);
  v34 = v71[0];
  if ( !v71[0] )
  {
    sub_93FB40((__int64)&v73, 0);
    v34 = v71[0];
    goto LABEL_48;
  }
  v35 = v73;
  v36 = v74;
  v37 = &v73[4 * (unsigned int)v74];
  if ( v73 == v37 )
  {
LABEL_44:
    if ( (unsigned int)v74 >= (unsigned __int64)HIDWORD(v74) )
    {
      v49 = (unsigned int)v74 + 1LL;
      if ( HIDWORD(v74) < v49 )
      {
        sub_C8D5F0((__int64)&v73, v75, v49, 0x10u, v50, v51);
        v37 = &v73[4 * (unsigned int)v74];
      }
      *(_QWORD *)v37 = 0;
      *((_QWORD *)v37 + 1) = v34;
      v34 = v71[0];
      LODWORD(v74) = v74 + 1;
    }
    else
    {
      if ( v37 )
      {
        *v37 = 0;
        *((_QWORD *)v37 + 1) = v34;
        v36 = v74;
        v34 = v71[0];
      }
      LODWORD(v74) = v36 + 1;
    }
LABEL_48:
    if ( !v34 )
      goto LABEL_3;
    goto LABEL_29;
  }
  while ( *v35 )
  {
    v35 += 4;
    if ( v37 == v35 )
      goto LABEL_44;
  }
  *((_QWORD *)v35 + 1) = v71[0];
LABEL_29:
  v13 = v34;
LABEL_2:
  sub_B91220((__int64)v71, v13);
LABEL_3:
  v14 = sub_B33F60((__int64)&v73, *(_QWORD *)(a1 + 96), a5, a6);
  v15 = *(_QWORD *)(a1 + 96);
  v16 = v14;
  v70 = 257;
  v55 = sub_AD64C0(v15, 3, 0);
  v17 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD))*v80 + 3))(v80, 26, v16, v55, 0);
  if ( !v17 )
  {
    v72 = 257;
    v17 = sub_B504D0(26, v16, v55, (__int64)v71, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _BYTE **, __int64, __int64))*v81 + 2))(v81, v17, &v67, v77, v78);
    if ( v73 != &v73[4 * (unsigned int)v74] )
    {
      v57 = a1;
      v22 = v73;
      v23 = &v73[4 * (unsigned int)v74];
      do
      {
        v24 = *((_QWORD *)v22 + 1);
        v25 = *v22;
        v22 += 4;
        sub_B99FD0(v17, v25, v24);
      }
      while ( v23 != v22 );
      a1 = v57;
    }
  }
  v18 = *(_QWORD *)(a1 + 96);
  v72 = 257;
  v19 = sub_94BCF0(&v73, a4, v18, (__int64)v71);
  v56 = v19;
  if ( a8 )
  {
    v72 = 257;
    if ( a9 )
    {
      v68 = v17;
      v67 = v19;
      v20 = sub_BCB2D0(v79);
      v69 = sub_ACD640(v20, a9, 0);
      v21 = sub_921880(
              &v73,
              *(_QWORD *)(32LL * a7 + a1 + 920),
              *(_QWORD *)(32LL * a7 + a1 + 928),
              (int)&v67,
              3,
              (__int64)v71,
              0);
    }
    else
    {
      v67 = v19;
      v68 = v17;
      v21 = sub_921880(
              &v73,
              *(_QWORD *)(32LL * a7 + a1 + 904),
              *(_QWORD *)(32LL * a7 + a1 + 912),
              (int)&v67,
              2,
              (__int64)v71,
              0);
    }
    if ( *(_BYTE *)(a10 + 8) )
    {
      v38 = *(unsigned int *)(a10 + 24);
      if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(a10 + 28) )
      {
        v62 = v21;
        sub_C8D5F0(a10 + 16, (const void *)(a10 + 32), v38 + 1, 8u, v38 + 1, v51);
        v38 = *(unsigned int *)(a10 + 24);
        v21 = v62;
      }
      *(_QWORD *)(*(_QWORD *)(a10 + 16) + 8 * v38) = v21;
      ++*(_DWORD *)(a10 + 24);
    }
  }
  else
  {
    v70 = 257;
    v26 = sub_AD64C0(*(_QWORD *)(a1 + 96), 1, 0);
    v27 = (_BYTE *)(*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD, _QWORD))*v80 + 4))(
                     v80,
                     15,
                     v17,
                     v26,
                     0,
                     0);
    if ( !v27 )
    {
      v72 = 257;
      v59 = sub_B504D0(15, v17, v26, (__int64)v71, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _BYTE **, __int64, __int64))*v81 + 2))(v81, v59, &v67, v77, v78);
      v27 = (_BYTE *)v59;
      if ( v73 != &v73[4 * (unsigned int)v74] )
      {
        v60 = v17;
        v44 = (__int64)v27;
        v52 = a1;
        v45 = v73;
        v46 = &v73[4 * (unsigned int)v74];
        do
        {
          v47 = *((_QWORD *)v45 + 1);
          v48 = *v45;
          v45 += 4;
          sub_B99FD0(v44, v48, v47);
        }
        while ( v46 != v45 );
        v27 = (_BYTE *)v44;
        a1 = v52;
        v17 = v60;
      }
    }
    v70 = 257;
    v28 = *(_QWORD *)(a4 + 8);
    v66 = 257;
    v29 = sub_929C50(&v73, v56, v27, (__int64)v65, 0, 0);
    if ( v28 == *(_QWORD *)(v29 + 8) )
    {
      v30 = v29;
    }
    else
    {
      v58 = v29;
      v30 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v80 + 15))(v80, 48, v29, v28);
      if ( !v30 )
      {
        v72 = 257;
        v30 = sub_B51D30(48, v58, v28, (__int64)v71, 0, 0);
        if ( (unsigned __int8)sub_920620(v30) )
        {
          v39 = v83;
          if ( v82 )
            sub_B99FD0(v30, 3u, v82);
          sub_B45150(v30, v39);
        }
        (*((void (__fastcall **)(void **, __int64, _BYTE **, __int64, __int64))*v81 + 2))(v81, v30, &v67, v77, v78);
        v40 = v73;
        v41 = &v73[4 * (unsigned int)v74];
        if ( v73 != v41 )
        {
          do
          {
            v42 = *((_QWORD *)v40 + 1);
            v43 = *v40;
            v40 += 4;
            sub_B99FD0(v30, v43, v42);
          }
          while ( v41 != v40 );
        }
      }
    }
    sub_23E39A0(a1, a2, a3, a4, 0, 8u, a7, v17, 0, a9, a10);
    sub_23E39A0(a1, a2, a3, v30, 0, 8u, a7, v17, 0, a9, a10);
  }
  nullsub_61();
  v88 = &unk_49DA100;
  nullsub_63();
  if ( v73 != (unsigned int *)v75 )
    _libc_free((unsigned __int64)v73);
}
