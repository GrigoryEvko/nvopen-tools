// Function: sub_1868880
// Address: 0x1868880
//
__int64 __fastcall sub_1868880(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r14
  unsigned __int8 v11; // al
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rcx
  int v15; // r8d
  int v16; // r9d
  double v17; // xmm4_8
  double v18; // xmm5_8
  char v19; // al
  __int64 ***v20; // rax
  __int64 **v21; // r13
  __int64 v22; // r12
  __int64 v23; // r12
  __int64 **v24; // r15
  unsigned __int64 v25; // r13
  __int64 v26; // r8
  int v27; // r9d
  __int64 v28; // rax
  __int64 *v29; // rdx
  int v30; // r14d
  __int64 v31; // rdi
  unsigned __int64 v32; // r15
  int v33; // ebx
  __int64 *v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdi
  unsigned __int8 v38; // al
  __int64 *v39; // rcx
  __int64 v40; // r13
  __int64 v41; // rax
  __int64 v42; // rbx
  __int64 *v43; // rdi
  __int64 *v44; // r15
  unsigned __int64 v45; // rax
  double v46; // xmm4_8
  double v47; // xmm5_8
  unsigned __int8 v48; // dl
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // r12
  bool v51; // zf
  unsigned __int64 v52; // rax
  __int64 v53; // r14
  _QWORD *v54; // rax
  double v55; // xmm4_8
  double v56; // xmm5_8
  _QWORD *v57; // r13
  __int64 v58; // rax
  int v60; // r8d
  int v61; // r9d
  __int64 v62; // r12
  __int64 v63; // rax
  __int64 v64; // [rsp+8h] [rbp-C8h]
  __int64 v65; // [rsp+18h] [rbp-B8h]
  __int64 v66; // [rsp+28h] [rbp-A8h]
  __int64 v67; // [rsp+30h] [rbp-A0h]
  __int64 *v68; // [rsp+38h] [rbp-98h]
  char v69; // [rsp+45h] [rbp-8Bh]
  unsigned __int8 v70; // [rsp+46h] [rbp-8Ah]
  unsigned __int8 v71; // [rsp+47h] [rbp-89h]
  __int64 v72; // [rsp+48h] [rbp-88h]
  __int64 v73; // [rsp+48h] [rbp-88h]
  __int64 v74; // [rsp+48h] [rbp-88h]
  __int64 **v75; // [rsp+50h] [rbp-80h]
  __int64 **v76; // [rsp+50h] [rbp-80h]
  __int64 v77; // [rsp+58h] [rbp-78h]
  unsigned int v78; // [rsp+6Ch] [rbp-64h] BYREF
  __int64 *v79; // [rsp+70h] [rbp-60h] BYREF
  __int64 v80; // [rsp+78h] [rbp-58h]
  _BYTE v81[80]; // [rsp+80h] [rbp-50h] BYREF

  v66 = a2;
  v71 = 0;
  if ( !(unsigned __int8)sub_1636800(a1, (__int64 *)a2) )
  {
    v77 = a2 + 24;
    do
    {
      v10 = *(_QWORD *)(v66 + 32);
      if ( v10 == v77 )
        return v71;
      v11 = v71;
      v71 = 0;
      v70 = v11;
      do
      {
        v12 = v10 - 56;
        if ( !v10 )
          v12 = 0;
        if ( !sub_15E4F60(v12) )
        {
          sub_159D9E0(v12);
          if ( (*(_BYTE *)(v12 + 32) & 0xFu) - 7 <= 1 )
            v71 |= sub_1868550(v12, (_BYTE *)a2, a3, a4, a5, a6, v17, v18, a9, a10, v13, v14, v15, v16);
          if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v12 + 24) + 16LL) + 8LL) )
          {
            sub_15E4B50(v12);
            if ( !v19 )
            {
              a2 = 18;
              v69 = sub_1560180(v12 + 112, 18);
              if ( !v69 )
              {
                v79 = (__int64 *)v81;
                v80 = 0x400000000LL;
                v20 = *(__int64 ****)(*(_QWORD *)(v12 + 24) + 16LL);
                v21 = *v20;
                if ( *((_BYTE *)*v20 + 8) == 13 )
                {
                  v22 = *((unsigned int *)v21 + 3);
                  if ( (_DWORD)v22 )
                  {
                    v23 = 8 * v22;
                    v24 = *v20;
                    v25 = 0;
                    do
                    {
                      v26 = sub_1599EF0((__int64 **)v24[2][v25 / 8]);
                      v28 = (unsigned int)v80;
                      if ( (unsigned int)v80 >= HIDWORD(v80) )
                      {
                        v74 = v26;
                        sub_16CD150((__int64)&v79, v81, 0, 8, v26, v27);
                        v28 = (unsigned int)v80;
                        v26 = v74;
                      }
                      v25 += 8LL;
                      v79[v28] = v26;
                      LODWORD(v80) = v80 + 1;
                    }
                    while ( v23 != v25 );
                    v21 = v24;
                  }
                }
                else
                {
                  v62 = sub_1599EF0(*v20);
                  v63 = (unsigned int)v80;
                  if ( (unsigned int)v80 >= HIDWORD(v80) )
                  {
                    sub_16CD150((__int64)&v79, v81, 0, 8, v60, v61);
                    v63 = (unsigned int)v80;
                  }
                  v21 = 0;
                  v79[v63] = v62;
                  LODWORD(v80) = v80 + 1;
                }
                a2 = v12 + 72;
                v29 = v79;
                v67 = v12 + 72;
                v72 = *(_QWORD *)(v12 + 80);
                if ( v72 != v12 + 72 )
                {
                  v64 = v12;
                  v75 = v21;
                  v68 = v79;
                  v65 = v10;
                  v30 = 0;
                  while ( 1 )
                  {
                    v31 = v72 - 24;
                    if ( !v72 )
                      v31 = 0;
                    v32 = sub_157EBA0(v31);
                    if ( *(_BYTE *)(v32 + 16) == 25 )
                    {
                      v33 = v80;
                      v78 = 0;
                      if ( (_DWORD)v80 )
                        break;
                    }
LABEL_42:
                    v72 = *(_QWORD *)(v72 + 8);
                    if ( v67 == v72 )
                    {
                      v12 = v64;
                      v10 = v65;
                      v21 = v75;
                      v29 = v68;
                      goto LABEL_44;
                    }
                  }
                  v34 = v68;
                  v35 = 0;
                  while ( 2 )
                  {
                    v40 = v34[v35];
                    v39 = v34;
                    if ( !v40 )
                    {
LABEL_36:
                      v35 = v78 + 1;
                      v78 = v35;
                      if ( v33 == (_DWORD)v35 )
                      {
                        v68 = v34;
                        goto LABEL_42;
                      }
                      continue;
                    }
                    break;
                  }
                  v41 = *(_DWORD *)(v32 + 20) & 0xFFFFFFF;
                  a2 = 4 * v41;
                  v37 = *(_QWORD *)(v32 - 24 * v41);
                  if ( v75 )
                  {
                    a2 = (__int64)&v78;
                    v36 = sub_14AC030(v37, &v78, 1, 0);
                    v34 = v79;
                    v37 = v36;
                    if ( v36 )
                      goto LABEL_29;
                  }
                  else
                  {
                    if ( !v37 )
                    {
LABEL_34:
                      ++v30;
                      v39[v78] = 0;
                      if ( v30 == (_DWORD)v80 )
                      {
                        v10 = v65;
                        v43 = v79;
                        goto LABEL_71;
                      }
                      v34 = v79;
                      goto LABEL_36;
                    }
LABEL_29:
                    v38 = *(_BYTE *)(v37 + 16);
                    if ( v38 == 9 )
                      goto LABEL_36;
                    if ( v38 <= 0x11u )
                    {
                      if ( *(_BYTE *)(v40 + 16) == 9 )
                      {
                        v34[v78] = v37;
                        v34 = v79;
                        goto LABEL_36;
                      }
                      if ( v40 == v37 )
                        goto LABEL_36;
                    }
                  }
                  v39 = v34;
                  goto LABEL_34;
                }
LABEL_44:
                v42 = *(_QWORD *)(v12 + 8);
                v43 = v29;
                if ( v42 )
                {
                  v73 = v10;
                  v44 = v29;
                  v76 = v21;
                  while ( 1 )
                  {
                    v45 = (unsigned __int64)sub_1648700(v42);
                    v48 = *(_BYTE *)(v45 + 16);
                    if ( v48 <= 0x17u )
                      goto LABEL_47;
                    if ( v48 == 78 )
                    {
                      v49 = v45 | 4;
LABEL_51:
                      v50 = v49 & 0xFFFFFFFFFFFFFFF8LL;
                      if ( (v49 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                        goto LABEL_47;
                      v51 = (v49 & 4) == 0;
                      v52 = v50 - 24;
                      if ( v51 )
                        v52 = v50 - 72;
                      if ( v52 != v42 )
                        goto LABEL_47;
                      v53 = *(_QWORD *)(v50 + 8);
                      if ( !v53 )
                        goto LABEL_47;
                      if ( !v76 )
                      {
                        a2 = *v44;
                        if ( *(_BYTE *)(*v44 + 16) == 17 )
                          a2 = *(_QWORD *)(v50
                                         + 24
                                         * (*(unsigned int *)(a2 + 32)
                                          - (unsigned __int64)(*(_DWORD *)(v50 + 20) & 0xFFFFFFF)));
                        sub_164D160(v50, a2, a3, a4, a5, a6, v46, v47, a9, a10);
                        v69 = 1;
                        v44 = v79;
                        goto LABEL_47;
                      }
                      do
                      {
                        v54 = sub_1648700(v53);
                        v53 = *(_QWORD *)(v53 + 8);
                        v57 = v54;
                        if ( *((_BYTE *)v54 + 16) == 86 )
                        {
                          v58 = *(int *)v54[7];
                          if ( (_DWORD)v58 != -1 )
                          {
                            a2 = v44[v58];
                            if ( a2 )
                            {
                              if ( *(_BYTE *)(a2 + 16) == 17 )
                                a2 = *(_QWORD *)(v50
                                               + 24
                                               * (*(unsigned int *)(a2 + 32)
                                                - (unsigned __int64)(*(_DWORD *)(v50 + 20) & 0xFFFFFFF)));
                              sub_164D160((__int64)v57, a2, a3, a4, a5, a6, v55, v56, a9, a10);
                              sub_15F20C0(v57);
                              v44 = v79;
                            }
                          }
                        }
                      }
                      while ( v53 );
                      v42 = *(_QWORD *)(v42 + 8);
                      v69 = 1;
                      if ( !v42 )
                      {
LABEL_65:
                        v10 = v73;
                        v43 = v44;
                        v70 |= v69;
                        break;
                      }
                    }
                    else
                    {
                      if ( v48 == 29 )
                      {
                        v49 = v45 & 0xFFFFFFFFFFFFFFFBLL;
                        goto LABEL_51;
                      }
LABEL_47:
                      v42 = *(_QWORD *)(v42 + 8);
                      if ( !v42 )
                        goto LABEL_65;
                    }
                  }
                }
LABEL_71:
                if ( v43 != (__int64 *)v81 )
                  _libc_free((unsigned __int64)v43);
              }
            }
          }
        }
        v10 = *(_QWORD *)(v10 + 8);
      }
      while ( v10 != v77 );
    }
    while ( v71 );
    return v70;
  }
  return v71;
}
