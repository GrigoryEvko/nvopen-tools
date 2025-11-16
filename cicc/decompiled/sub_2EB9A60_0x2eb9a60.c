// Function: sub_2EB9A60
// Address: 0x2eb9a60
//
_QWORD *__fastcall sub_2EB9A60(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rax
  int v14; // ebx
  __int64 v15; // r14
  int v16; // r15d
  __int64 v17; // rdx
  __int64 v18; // r9
  int v19; // eax
  unsigned int v20; // eax
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // r15
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r12
  __int64 v33; // rdx
  __int64 v34; // r13
  __int64 v35; // rsi
  _QWORD *v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // r8
  __int64 v40; // r9
  unsigned int v41; // eax
  unsigned int v42; // ebx
  __int64 v43; // rcx
  __int64 v44; // rax
  unsigned int v45; // r11d
  __int64 *v46; // r12
  __int64 v47; // r9
  int v48; // r14d
  __int64 v49; // rax
  char *v50; // rbx
  char *v51; // r14
  unsigned __int64 v52; // rdi
  _QWORD *v53; // r10
  unsigned __int64 v54; // r14
  _QWORD *v55; // rsi
  unsigned int v56; // r9d
  __int64 v57; // rdx
  __int64 *v58; // rax
  unsigned int v59; // r11d
  _BYTE *v60; // rbx
  unsigned __int64 v61; // r12
  unsigned __int64 v62; // rdi
  __int64 v64; // [rsp+18h] [rbp-2148h]
  int v67; // [rsp+40h] [rbp-2120h]
  __int64 v68; // [rsp+40h] [rbp-2120h]
  __int64 v69; // [rsp+48h] [rbp-2118h]
  __int64 v70; // [rsp+48h] [rbp-2118h]
  unsigned int v71; // [rsp+48h] [rbp-2118h]
  __int64 v72; // [rsp+48h] [rbp-2118h]
  unsigned int v73; // [rsp+58h] [rbp-2108h]
  __int64 v74[4]; // [rsp+60h] [rbp-2100h] BYREF
  __int128 v75; // [rsp+80h] [rbp-20E0h] BYREF
  __int128 v76; // [rsp+90h] [rbp-20D0h] BYREF
  __int64 v77; // [rsp+A0h] [rbp-20C0h]
  _QWORD *v78; // [rsp+D0h] [rbp-2090h] BYREF
  __int64 v79; // [rsp+D8h] [rbp-2088h]
  _QWORD v80[64]; // [rsp+E0h] [rbp-2080h] BYREF
  _BYTE *v81; // [rsp+2E0h] [rbp-1E80h]
  __int64 v82; // [rsp+2E8h] [rbp-1E78h]
  _BYTE v83[3584]; // [rsp+2F0h] [rbp-1E70h] BYREF
  __int64 v84; // [rsp+10F0h] [rbp-1070h]
  __int64 *v85; // [rsp+1100h] [rbp-1060h] BYREF
  __int64 v86; // [rsp+1108h] [rbp-1058h]
  __int64 v87; // [rsp+1110h] [rbp-1050h] BYREF
  char *v88[2]; // [rsp+1118h] [rbp-1048h] BYREF
  _BYTE v89[488]; // [rsp+1128h] [rbp-1038h] BYREF
  char *v90; // [rsp+1310h] [rbp-E50h]
  __int64 v91; // [rsp+1318h] [rbp-E48h]
  char v92; // [rsp+1320h] [rbp-E40h] BYREF
  __int64 v93; // [rsp+2120h] [rbp-40h]

  *a1 = a1 + 2;
  a1[1] = 0x400000000LL;
  v78 = v80;
  v79 = 0x4000000001LL;
  v81 = v83;
  v84 = a3;
  v80[0] = 0;
  v82 = 0x4000000000LL;
  v7 = sub_2EB5B40((__int64)&v78, 0, a3, a4, a5, a6);
  *(_QWORD *)(v7 + 8) = 0x100000001LL;
  *(_DWORD *)v7 = 1;
  sub_2E6D5A0((__int64)&v78, 0, 0x100000001LL, v8, v9, v10);
  v13 = *(_QWORD *)(a2 + 128);
  v69 = v13 + 320;
  if ( *(_QWORD *)(v13 + 328) != v13 + 320 )
  {
    v14 = 0;
    v73 = 1;
    v15 = *(_QWORD *)(v13 + 328);
    do
    {
      while ( 1 )
      {
        v16 = v14;
        sub_2EB5530(&v85, v15, a3, v11, v12);
        ++v14;
        v19 = v86;
        if ( v85 != &v87 )
        {
          v67 = v86;
          _libc_free((unsigned __int64)v85);
          v19 = v67;
        }
        if ( !v19 )
          break;
        v15 = *(_QWORD *)(v15 + 8);
        if ( v69 == v15 )
          goto LABEL_8;
      }
      sub_2E6D5A0((__int64)a1, v15, v17, v11, v12, v18);
      v20 = sub_2EB8890((__int64)&v78, v15, v73, (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_2EB2D60, 1, 0);
      v15 = *(_QWORD *)(v15 + 8);
      v73 = v20;
    }
    while ( v69 != v15 );
LABEL_8:
    if ( v73 != v16 + 2 )
    {
      v77 = 0;
      v75 = 0;
      v74[1] = a2;
      v21 = *(_QWORD *)(a2 + 128);
      v76 = 0;
      v22 = *(_QWORD *)(v21 + 328);
      v23 = v21 + 320;
      v74[0] = (__int64)&v75;
      v74[2] = (__int64)&v78;
      v70 = v21 + 320;
      if ( v21 + 320 != v22 )
      {
        do
        {
          while ( *(_DWORD *)sub_2EB5B40((__int64)&v78, v22, v23, v11, v12, v18) )
          {
            v22 = *(_QWORD *)(v22 + 8);
            if ( v70 == v22 )
              goto LABEL_23;
          }
          if ( !(_BYTE)v77 )
            sub_2EB75A0(v74);
          v24 = (unsigned int)sub_2EB9660(
                                (__int64)&v78,
                                v22,
                                v73,
                                (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_2EB2D60,
                                v73,
                                (__int64 *)&v75);
          v68 = v78[v24];
          sub_2E6D5A0((__int64)a1, v68, v25, v26, v27, v28);
          if ( (unsigned int)v24 > v73 )
          {
            v64 = v22;
            v32 = 8 * v24;
            v33 = v24 - ((unsigned int)v24 + ~v73);
            v34 = 8 * v33;
            while ( 1 )
            {
              v35 = v78[(unsigned __int64)v32 / 8];
              v88[0] = v89;
              v85 = 0;
              v86 = 0;
              v87 = 0;
              v88[1] = (char *)0x400000000LL;
              v36 = (_QWORD *)sub_2EB5B40((__int64)&v78, v35, v33, v29, v30, v31);
              *v36 = v85;
              v36[1] = v86;
              v37 = v87;
              v36[2] = v87;
              sub_2EB32F0((__int64)(v36 + 3), v88, v38, v37, v39, v40);
              if ( v88[0] != v89 )
                _libc_free((unsigned __int64)v88[0]);
              LODWORD(v79) = v79 - 1;
              if ( v34 == v32 )
                break;
              v32 -= 8;
            }
            v22 = v64;
          }
          v41 = sub_2EB8890((__int64)&v78, v68, v73, (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_2EB2D60, 1, 0);
          v22 = *(_QWORD *)(v22 + 8);
          v73 = v41;
        }
        while ( v70 != v22 );
LABEL_23:
        if ( (_BYTE)v77 )
        {
          LOBYTE(v77) = 0;
          sub_C7D6A0(*((__int64 *)&v75 + 1), 16LL * DWORD2(v76), 8);
        }
      }
      v42 = 0;
      v85 = &v87;
      v86 = 0x4000000001LL;
      v43 = *((unsigned int *)a1 + 2);
      v90 = &v92;
      v91 = 0x4000000000LL;
      v87 = 0;
      v93 = a3;
      v44 = 0;
      if ( (_DWORD)v43 )
      {
        do
        {
          while ( 1 )
          {
            v46 = (__int64 *)(*a1 + 8 * v44);
            sub_2EB5530(&v75, *v46, a3, v43, v12);
            v48 = DWORD2(v75);
            if ( (__int128 *)v75 != &v76 )
              _libc_free(v75);
            if ( !v48 )
              break;
            v49 = 0;
            LODWORD(v86) = 0;
            if ( !HIDWORD(v86) )
            {
              sub_C8D5F0((__int64)&v85, &v87, 1u, 8u, v12, v47);
              v49 = (unsigned int)v86;
            }
            v85[v49] = 0;
            LODWORD(v86) = v86 + 1;
            if ( v90 != &v90[56 * (unsigned int)v91] )
            {
              v71 = v42;
              v50 = &v90[56 * (unsigned int)v91];
              v51 = v90;
              do
              {
                v50 -= 56;
                v52 = *((_QWORD *)v50 + 3);
                if ( (char *)v52 != v50 + 40 )
                  _libc_free(v52);
              }
              while ( v51 != v50 );
              v42 = v71;
            }
            LODWORD(v91) = 0;
            if ( (unsigned int)sub_2EB9660(
                                 (__int64)&v85,
                                 *v46,
                                 0,
                                 (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_2EB2D60,
                                 0,
                                 0) <= 1 )
              break;
            v53 = (_QWORD *)*a1;
            LODWORD(v12) = 2;
            v54 = (unsigned __int64)v85;
            v72 = *((unsigned int *)a1 + 2);
            v55 = (_QWORD *)(*a1 + v72 * 8);
            while ( 1 )
            {
              *(_QWORD *)&v75 = *(_QWORD *)(v54 + 8LL * (unsigned int)v12);
              if ( v55 != sub_2EB3010(v53, (__int64)v55, (__int64 *)&v75) )
                break;
              v12 = (unsigned int)(v12 + 1);
              if ( v56 < (unsigned int)v12 )
                goto LABEL_28;
            }
            v57 = *v46;
            v58 = &v53[v72 - 1];
            v43 = *v58;
            *v46 = *v58;
            *v58 = v57;
            v59 = *((_DWORD *)a1 + 2) - 1;
            v44 = v42;
            *((_DWORD *)a1 + 2) = v59;
            if ( v59 <= v42 )
              goto LABEL_45;
          }
          v45 = *((_DWORD *)a1 + 2);
LABEL_28:
          v44 = ++v42;
        }
        while ( v45 > v42 );
      }
LABEL_45:
      sub_2EB40F0((__int64)&v85);
    }
  }
  v60 = v81;
  v61 = (unsigned __int64)&v81[56 * (unsigned int)v82];
  if ( v81 != (_BYTE *)v61 )
  {
    do
    {
      v61 -= 56LL;
      v62 = *(_QWORD *)(v61 + 24);
      if ( v62 != v61 + 40 )
        _libc_free(v62);
    }
    while ( v60 != (_BYTE *)v61 );
    v61 = (unsigned __int64)v81;
  }
  if ( (_BYTE *)v61 != v83 )
    _libc_free(v61);
  if ( v78 != v80 )
    _libc_free((unsigned __int64)v78);
  return a1;
}
