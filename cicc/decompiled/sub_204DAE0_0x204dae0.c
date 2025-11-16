// Function: sub_204DAE0
// Address: 0x204dae0
//
void __fastcall sub_204DAE0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  __int64 v10; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 (*v14)(void); // rax
  unsigned int v15; // eax
  unsigned __int16 ***v16; // rdx
  unsigned __int8 v17; // bl
  unsigned __int16 ***v18; // r12
  int v19; // ebx
  int v20; // r9d
  __int64 v21; // r8
  unsigned __int8 v22; // al
  size_t i; // r14
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rcx
  int v29; // r8d
  int v30; // r9d
  __int64 v31; // rdx
  __int64 v32; // rcx
  int v33; // r8d
  int v34; // r9d
  __int64 v35; // rdx
  __int64 v36; // rcx
  int v37; // r8d
  int v38; // r9d
  __int64 v39; // rdx
  __int64 v40; // rcx
  int v41; // r8d
  int v42; // r9d
  int v43; // ebx
  char v44; // di
  unsigned int v45; // eax
  bool v46; // zf
  unsigned __int8 *v47; // rax
  unsigned int v48; // r14d
  __int64 v49; // rax
  unsigned int v50; // eax
  unsigned __int16 *v51; // rdx
  int v52; // ecx
  unsigned __int16 *v53; // r13
  __int64 v54; // r12
  int v55; // ebx
  char *v56; // rax
  char v57; // r10
  char v58; // dl
  unsigned int v59; // eax
  int v60; // esi
  unsigned int v61; // eax
  int v62; // ecx
  unsigned __int8 v63; // r10
  int v64; // esi
  __int64 v65; // rax
  int v66; // edx
  int v67; // edx
  __int64 v68; // [rsp+8h] [rbp-188h]
  unsigned __int8 v69; // [rsp+8h] [rbp-188h]
  __int64 v71; // [rsp+10h] [rbp-180h]
  __int64 v72; // [rsp+18h] [rbp-178h]
  unsigned int v73; // [rsp+18h] [rbp-178h]
  __int64 v74; // [rsp+20h] [rbp-170h]
  unsigned __int8 v76; // [rsp+28h] [rbp-168h]
  unsigned __int8 v77; // [rsp+28h] [rbp-168h]
  unsigned int v78; // [rsp+30h] [rbp-160h]
  unsigned __int8 v79; // [rsp+30h] [rbp-160h]
  char v80; // [rsp+6Bh] [rbp-125h] BYREF
  unsigned int v81; // [rsp+6Ch] [rbp-124h] BYREF
  __int64 v82; // [rsp+70h] [rbp-120h] BYREF
  __int64 v83; // [rsp+78h] [rbp-118h]
  __int64 v84; // [rsp+80h] [rbp-110h] BYREF
  __int64 v85; // [rsp+88h] [rbp-108h]
  _BYTE *v86; // [rsp+90h] [rbp-100h] BYREF
  __int64 v87; // [rsp+98h] [rbp-F8h]
  _BYTE v88[16]; // [rsp+A0h] [rbp-F0h] BYREF
  char *v89; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 v90; // [rsp+B8h] [rbp-D8h]
  __int64 v91; // [rsp+C0h] [rbp-D0h] BYREF
  char *v92; // [rsp+100h] [rbp-90h] BYREF
  char v93; // [rsp+110h] [rbp-80h] BYREF
  char *v94; // [rsp+118h] [rbp-78h] BYREF
  char v95; // [rsp+128h] [rbp-68h] BYREF
  char *v96; // [rsp+138h] [rbp-58h] BYREF
  char v97; // [rsp+148h] [rbp-48h] BYREF
  int v98; // [rsp+158h] [rbp-38h]
  char v99; // [rsp+15Ch] [rbp-34h]

  v10 = 0;
  v12 = a1[6];
  v86 = v88;
  v74 = v12;
  v13 = a1[4];
  v87 = 0x400000000LL;
  v72 = v13;
  v14 = *(__int64 (**)(void))(**(_QWORD **)(v13 + 16) + 112LL);
  if ( v14 != sub_1D00B10 )
    v10 = v14();
  v15 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)a2 + 1392LL))(
          a2,
          v10,
          *(_QWORD *)(a5 + 192),
          *(_QWORD *)(a5 + 200),
          *(unsigned __int8 *)(a5 + 240));
  v17 = *(_BYTE *)(a4 + 240);
  v78 = v15;
  v18 = v16;
  if ( v17 == 1 )
    goto LABEL_38;
  if ( *(_DWORD *)a4 <= 1u && v16 )
  {
    v56 = *(char **)(*(_QWORD *)(v10 + 280)
                   + 24LL
                   * (*((unsigned __int16 *)*v16 + 12)
                    + *(_DWORD *)(v10 + 288)
                    * (unsigned int)((__int64)(*(_QWORD *)(v10 + 264) - *(_QWORD *)(v10 + 256)) >> 3))
                   + 16);
    v57 = *v56;
    if ( *v56 == 1 )
    {
      v64 = sub_2045180(1);
      if ( (unsigned int)sub_2045180(v17) == v64 )
      {
LABEL_74:
        if ( !v62 && !*(_BYTE *)(a4 + 10) )
        {
          v69 = v63;
          v65 = sub_1D309E0(a1, 158, a3, v63, 0, 0, a6, a7, a8, *(_OWORD *)(a4 + 248));
          v63 = v69;
          *(_QWORD *)(a4 + 248) = v65;
          *(_DWORD *)(a4 + 256) = v66;
        }
        *(_BYTE *)(a4 + 240) = v63;
        v17 = v63;
      }
    }
    else
    {
      v58 = *v56;
      while ( v17 != v58 )
      {
        v58 = *++v56;
        if ( v58 == 1 )
        {
          v60 = sub_2045180(v57);
          v61 = sub_2045180(v17);
          if ( v60 == v61 )
            goto LABEL_74;
          if ( ((unsigned __int8)(v63 - 14) <= 0x47u || (unsigned __int8)(v63 - 2) <= 5u)
            && ((unsigned __int8)(v17 - 8) <= 5u || (unsigned __int8)(v17 - 86) <= 0x17u) )
          {
            if ( v61 == 32 )
            {
              v17 = 5;
            }
            else if ( v61 > 0x20 )
            {
              v17 = 6;
              if ( v61 != 64 )
              {
                v17 = 0;
                if ( v61 == 128 )
                  v17 = 7;
              }
            }
            else
            {
              v17 = 3;
              if ( v61 != 8 )
              {
                v17 = 4;
                if ( v61 != 16 )
                  v17 = 2 * (v61 == 1);
              }
            }
            if ( !v62 )
            {
              *(_QWORD *)(a4 + 248) = sub_1D309E0(a1, 158, a3, v17, 0, 0, a6, a7, a8, *(_OWORD *)(a4 + 248));
              *(_DWORD *)(a4 + 256) = v67;
            }
            *(_BYTE *)(a4 + 240) = v17;
          }
          break;
        }
      }
    }
  }
  LOBYTE(v82) = v17;
  v83 = 0;
  if ( v17 )
  {
    v19 = *(unsigned __int8 *)(a2 + v17 + 1040);
  }
  else if ( sub_1F58D20((__int64)&v82) )
  {
    LOBYTE(v89) = 0;
    v90 = 0;
    LOBYTE(v81) = 0;
    v19 = sub_1F426C0(a2, v74, (unsigned int)v82, 0, (__int64)&v89, (unsigned int *)&v84, &v81);
  }
  else
  {
    v43 = sub_1F58D40((__int64)&v82);
    v84 = v82;
    v68 = v82;
    v85 = v83;
    v71 = v83;
    if ( sub_1F58D20((__int64)&v84) )
    {
      LOBYTE(v89) = 0;
      v90 = 0;
      v80 = 0;
      sub_1F426C0(a2, v74, (unsigned int)v84, v85, (__int64)&v89, &v81, &v80);
      v44 = v80;
    }
    else
    {
      sub_1F40D10((__int64)&v89, a2, v74, v68, v71);
      v44 = sub_1D5E9F0(a2, v74, (unsigned __int8)v90, v91);
    }
    v45 = sub_2045180(v44);
    v19 = (v45 + v43 - 1) / v45;
  }
  if ( (unsigned int)(*(_DWORD *)(*a1 + 504) - 34) <= 1 && *(_BYTE *)(a4 + 240) == 7 )
LABEL_38:
    v19 = 1;
  if ( !(unsigned __int8)sub_20B4290(a4) )
  {
    v76 = *(_BYTE *)(a4 + 240);
    v21 = v78;
    if ( v78 )
    {
      v47 = *(unsigned __int8 **)(*(_QWORD *)(v10 + 280)
                                + 24LL
                                * (*((unsigned __int16 *)*v18 + 12)
                                 + *(_DWORD *)(v10 + 288)
                                 * (unsigned int)((__int64)(*(_QWORD *)(v10 + 264) - *(_QWORD *)(v10 + 256)) >> 3))
                                + 16);
      if ( v76 == 1 )
      {
        v59 = *v47;
        v76 = v59;
        v48 = v59;
      }
      else
      {
        v48 = *v47;
      }
      v49 = (unsigned int)v87;
      if ( (unsigned int)v87 >= HIDWORD(v87) )
      {
        sub_16CD150((__int64)&v86, v88, 0, 4, v78, v20);
        v49 = (unsigned int)v87;
        LODWORD(v21) = v78;
      }
      *(_DWORD *)&v86[4 * v49] = v78;
      v50 = v87 + 1;
      LODWORD(v87) = v87 + 1;
      if ( v19 != 1 )
      {
        v51 = **v18;
        if ( v78 != *v51 )
        {
          do
          {
            v52 = v51[1];
            ++v51;
          }
          while ( (_DWORD)v21 != v52 );
        }
        v53 = v51 + 1;
        v54 = (__int64)&v51[v19 - 2 + 2];
        do
        {
          v55 = *v53;
          if ( v50 >= HIDWORD(v87) )
          {
            sub_16CD150((__int64)&v86, v88, 0, 4, v21, v20);
            v50 = v87;
          }
          ++v53;
          *(_DWORD *)&v86[4 * v50] = v55;
          v50 = v87 + 1;
          LODWORD(v87) = v87 + 1;
        }
        while ( v53 != (unsigned __int16 *)v54 );
      }
      BYTE4(v84) = 0;
      v25 = v48;
      v26 = v76;
    }
    else
    {
      if ( !v18 )
        goto LABEL_31;
      v22 = *(_BYTE *)(a4 + 240);
      v79 = **(_BYTE **)(*(_QWORD *)(v10 + 280)
                       + 24LL
                       * (*((unsigned __int16 *)*v18 + 12)
                        + *(_DWORD *)(v10 + 288)
                        * (unsigned int)((__int64)(*(_QWORD *)(v10 + 264) - *(_QWORD *)(v10 + 256)) >> 3))
                       + 16);
      if ( v76 == 1 )
        v22 = **(_BYTE **)(*(_QWORD *)(v10 + 280)
                         + 24LL
                         * (*((unsigned __int16 *)*v18 + 12)
                          + *(_DWORD *)(v10 + 288)
                          * (unsigned int)((__int64)(*(_QWORD *)(v10 + 264) - *(_QWORD *)(v10 + 256)) >> 3))
                         + 16);
      v77 = v22;
      for ( i = *(_QWORD *)(v72 + 40); v19; --v19 )
      {
        LODWORD(v21) = sub_1E6B9A0(i, (__int64)v18, (unsigned __int8 *)byte_3F871B3, 0, v21, v20);
        v24 = (unsigned int)v87;
        if ( (unsigned int)v87 >= HIDWORD(v87) )
        {
          v73 = v21;
          sub_16CD150((__int64)&v86, v88, 0, 4, v21, v20);
          v24 = (unsigned int)v87;
          v21 = v73;
        }
        *(_DWORD *)&v86[4 * v24] = v21;
        LODWORD(v87) = v87 + 1;
      }
      BYTE4(v84) = 0;
      v25 = v79;
      v26 = v77;
    }
    sub_204DA20((__int64)&v89, (__int64)&v86, v25, v26, 0, (unsigned int *)&v84);
    sub_20449C0(a4 + 264, &v89, v27, v28, v29, v30);
    sub_2044890(a4 + 344, &v92, v31, v32, v33, v34);
    sub_2044C40(a4 + 368, &v94, v35, v36, v37, v38);
    sub_2044C40(a4 + 400, &v96, v39, v40, v41, v42);
    if ( v99 )
    {
      v46 = *(_BYTE *)(a4 + 436) == 0;
      *(_DWORD *)(a4 + 432) = v98;
      if ( v46 )
        *(_BYTE *)(a4 + 436) = 1;
    }
    else if ( *(_BYTE *)(a4 + 436) )
    {
      *(_BYTE *)(a4 + 436) = 0;
    }
    if ( v96 != &v97 )
      _libc_free((unsigned __int64)v96);
    if ( v94 != &v95 )
      _libc_free((unsigned __int64)v94);
    if ( v92 != &v93 )
      _libc_free((unsigned __int64)v92);
    if ( v89 != (char *)&v91 )
      _libc_free((unsigned __int64)v89);
  }
LABEL_31:
  if ( v86 != v88 )
    _libc_free((unsigned __int64)v86);
}
