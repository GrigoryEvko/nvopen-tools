// Function: sub_17F5B20
// Address: 0x17f5b20
//
__int64 __fastcall sub_17F5B20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rbx
  int v10; // r8d
  __int64 v11; // r9
  __int64 v12; // rax
  unsigned __int8 *v13; // rsi
  __int64 v14; // rbx
  __int64 v15; // rax
  int v16; // r8d
  int v17; // r9d
  _QWORD *v18; // r15
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 **v21; // r15
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rdi
  _QWORD *v27; // r15
  __int64 v28; // rax
  __int64 **v29; // r10
  __int64 v30; // rdi
  __int64 v31; // rax
  _QWORD *v32; // r15
  __int64 v33; // rax
  __int64 v34; // r14
  _QWORD *v35; // rax
  __int64 v36; // r15
  __int64 v37; // r13
  __int128 v38; // rdi
  __int64 v39; // rcx
  __int64 v40; // rax
  unsigned __int8 *v41; // rsi
  __int64 v43; // rax
  unsigned __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rdx
  unsigned __int8 *v48; // rsi
  __int64 v49; // rax
  __int64 v50; // r9
  __int64 v51; // rax
  __int64 v52; // rsi
  __int64 v53; // rsi
  __int64 v54; // rdx
  unsigned __int8 *v55; // rsi
  __int64 v56; // rax
  unsigned __int64 v57; // rsi
  __int64 v58; // rax
  __int64 v59; // rsi
  __int64 v60; // rdx
  unsigned __int8 *v61; // rsi
  __int64 v62; // rax
  unsigned __int64 v63; // rsi
  __int64 v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rdx
  unsigned __int8 *v67; // rsi
  __int64 *v68; // [rsp+8h] [rbp-208h]
  __int64 v70; // [rsp+28h] [rbp-1E8h]
  __int64 **v71; // [rsp+28h] [rbp-1E8h]
  unsigned __int64 *v72; // [rsp+28h] [rbp-1E8h]
  __int64 v73; // [rsp+28h] [rbp-1E8h]
  __int64 v74; // [rsp+28h] [rbp-1E8h]
  __int64 v75; // [rsp+28h] [rbp-1E8h]
  __int64 v76; // [rsp+28h] [rbp-1E8h]
  unsigned __int64 *v77; // [rsp+28h] [rbp-1E8h]
  unsigned __int64 *v78; // [rsp+28h] [rbp-1E8h]
  unsigned __int8 *v79; // [rsp+38h] [rbp-1D8h] BYREF
  __int64 v80[2]; // [rsp+40h] [rbp-1D0h] BYREF
  __int16 v81; // [rsp+50h] [rbp-1C0h]
  unsigned __int8 *v82[2]; // [rsp+60h] [rbp-1B0h] BYREF
  __int16 v83; // [rsp+70h] [rbp-1A0h]
  unsigned __int8 *v84; // [rsp+80h] [rbp-190h] BYREF
  __int64 v85; // [rsp+88h] [rbp-188h]
  unsigned __int64 *v86; // [rsp+90h] [rbp-180h]
  __int64 v87; // [rsp+98h] [rbp-178h]
  __int64 v88; // [rsp+A0h] [rbp-170h]
  int v89; // [rsp+A8h] [rbp-168h]
  __int64 v90; // [rsp+B0h] [rbp-160h]
  __int64 v91; // [rsp+B8h] [rbp-158h]
  _BYTE *v92; // [rsp+D0h] [rbp-140h] BYREF
  __int64 v93; // [rsp+D8h] [rbp-138h]
  _BYTE v94[304]; // [rsp+E0h] [rbp-130h] BYREF

  v7 = *(_QWORD *)(a2 + 80);
  v92 = v94;
  v93 = 0x2000000000LL;
  if ( v7 )
    v7 -= 24;
  v8 = sub_157EE30(v7);
  if ( !v8 )
  {
    v84 = 0;
    v86 = 0;
    v87 = sub_16498A0(0);
    v88 = 0;
    v89 = 0;
    v90 = 0;
    v91 = 0;
    v85 = 0;
    BUG();
  }
  v9 = v8;
  v88 = 0;
  v87 = sub_16498A0(v8 - 24);
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v12 = *(_QWORD *)(v9 + 16);
  v13 = *(unsigned __int8 **)(v9 + 24);
  v86 = (unsigned __int64 *)v9;
  v84 = 0;
  v85 = v12;
  v82[0] = v13;
  if ( v13 )
  {
    sub_1623A60((__int64)v82, (__int64)v13, 2);
    if ( v84 )
      sub_161E7C0((__int64)&v84, (__int64)v84);
    v84 = v82[0];
    if ( v82[0] )
      sub_1623210((__int64)v82, v82[0], (__int64)&v84);
  }
  if ( a4 )
  {
    v14 = 0;
    do
    {
      while ( 1 )
      {
        v24 = *(_QWORD *)(a2 + 80);
        v25 = *(_QWORD *)(a1 + 304);
        v26 = *(_QWORD *)(a3 + 8 * v14);
        if ( v24 )
          v24 -= 24;
        if ( v26 != v24 )
          break;
        v27 = (_QWORD *)a2;
        v81 = 257;
        if ( *(_QWORD *)a2 != v25 )
        {
          if ( *(_BYTE *)(a2 + 16) > 0x10u )
          {
            v83 = 257;
            v62 = sub_15FDFF0(a2, v25, (__int64)v82, 0);
            v27 = (_QWORD *)v62;
            if ( v85 )
            {
              v78 = v86;
              sub_157E9D0(v85 + 40, v62);
              v63 = *v78;
              v64 = v27[3] & 7LL;
              v27[4] = v78;
              v63 &= 0xFFFFFFFFFFFFFFF8LL;
              v27[3] = v63 | v64;
              *(_QWORD *)(v63 + 8) = v27 + 3;
              *v78 = *v78 & 7 | (unsigned __int64)(v27 + 3);
            }
            sub_164B780((__int64)v27, v80);
            if ( v84 )
            {
              v79 = v84;
              sub_1623A60((__int64)&v79, (__int64)v84, 2);
              v65 = v27[6];
              v66 = (__int64)(v27 + 6);
              if ( v65 )
              {
                sub_161E7C0((__int64)(v27 + 6), v65);
                v66 = (__int64)(v27 + 6);
              }
              v67 = v79;
              v27[6] = v79;
              if ( v67 )
              {
                sub_1623210((__int64)&v79, v67, v66);
                v28 = (unsigned int)v93;
                if ( (unsigned int)v93 < HIDWORD(v93) )
                  goto LABEL_27;
                goto LABEL_68;
              }
            }
          }
          else
          {
            v27 = (_QWORD *)sub_15A4A70((__int64 ***)a2, v25);
          }
        }
        v28 = (unsigned int)v93;
        if ( (unsigned int)v93 < HIDWORD(v93) )
          goto LABEL_27;
LABEL_68:
        sub_16CD150((__int64)&v92, v94, 0, 8, v10, v11);
        v28 = (unsigned int)v93;
LABEL_27:
        *(_QWORD *)&v92[8 * v28] = v27;
        v29 = *(__int64 ***)(a1 + 304);
        v30 = *(_QWORD *)(a1 + 296);
        LODWORD(v93) = v93 + 1;
        v71 = v29;
        v81 = 257;
        v31 = sub_15A0680(v30, 1, 0);
        v32 = (_QWORD *)v31;
        if ( v71 == *(__int64 ***)v31 )
          goto LABEL_30;
        if ( *(_BYTE *)(v31 + 16) <= 0x10u )
        {
          v32 = (_QWORD *)sub_15A46C0(46, (__int64 ***)v31, v71, 0);
LABEL_30:
          v33 = (unsigned int)v93;
          if ( (unsigned int)v93 >= HIDWORD(v93) )
            goto LABEL_60;
          goto LABEL_31;
        }
        v83 = 257;
        v56 = sub_15FDBD0(46, v31, (__int64)v71, (__int64)v82, 0);
        v32 = (_QWORD *)v56;
        if ( v85 )
        {
          v77 = v86;
          sub_157E9D0(v85 + 40, v56);
          v57 = *v77;
          v58 = v32[3] & 7LL;
          v32[4] = v77;
          v57 &= 0xFFFFFFFFFFFFFFF8LL;
          v32[3] = v57 | v58;
          *(_QWORD *)(v57 + 8) = v32 + 3;
          *v77 = *v77 & 7 | (unsigned __int64)(v32 + 3);
        }
        sub_164B780((__int64)v32, v80);
        if ( !v84 )
          goto LABEL_30;
        v79 = v84;
        sub_1623A60((__int64)&v79, (__int64)v84, 2);
        v59 = v32[6];
        v60 = (__int64)(v32 + 6);
        if ( v59 )
        {
          sub_161E7C0((__int64)(v32 + 6), v59);
          v60 = (__int64)(v32 + 6);
        }
        v61 = v79;
        v32[6] = v79;
        if ( !v61 )
          goto LABEL_30;
        sub_1623210((__int64)&v79, v61, v60);
        v33 = (unsigned int)v93;
        if ( (unsigned int)v93 >= HIDWORD(v93) )
        {
LABEL_60:
          sub_16CD150((__int64)&v92, v94, 0, 8, v10, v11);
          v33 = (unsigned int)v93;
        }
LABEL_31:
        ++v14;
        *(_QWORD *)&v92[8 * v33] = v32;
        LODWORD(v93) = v93 + 1;
        if ( v14 == a4 )
          goto LABEL_32;
      }
      v70 = *(_QWORD *)(a1 + 304);
      v81 = 257;
      v15 = sub_159BF40(v26);
      v18 = (_QWORD *)v15;
      if ( *(_QWORD *)v15 != v70 )
      {
        if ( *(_BYTE *)(v15 + 16) > 0x10u )
        {
          v83 = 257;
          v43 = sub_15FDFF0(v15, v70, (__int64)v82, 0);
          v18 = (_QWORD *)v43;
          if ( v85 )
          {
            v72 = v86;
            sub_157E9D0(v85 + 40, v43);
            v44 = *v72;
            v45 = v18[3] & 7LL;
            v18[4] = v72;
            v44 &= 0xFFFFFFFFFFFFFFF8LL;
            v18[3] = v44 | v45;
            *(_QWORD *)(v44 + 8) = v18 + 3;
            *v72 = *v72 & 7 | (unsigned __int64)(v18 + 3);
          }
          sub_164B780((__int64)v18, v80);
          if ( v84 )
          {
            v79 = v84;
            sub_1623A60((__int64)&v79, (__int64)v84, 2);
            v46 = v18[6];
            v47 = (__int64)(v18 + 6);
            if ( v46 )
            {
              sub_161E7C0((__int64)(v18 + 6), v46);
              v47 = (__int64)(v18 + 6);
            }
            v48 = v79;
            v18[6] = v79;
            if ( v48 )
            {
              sub_1623210((__int64)&v79, v48, v47);
              v19 = (unsigned int)v93;
              if ( (unsigned int)v93 < HIDWORD(v93) )
                goto LABEL_15;
              goto LABEL_44;
            }
          }
        }
        else
        {
          v18 = (_QWORD *)sub_15A4A70((__int64 ***)v15, v70);
        }
      }
      v19 = (unsigned int)v93;
      if ( (unsigned int)v93 < HIDWORD(v93) )
        goto LABEL_15;
LABEL_44:
      sub_16CD150((__int64)&v92, v94, 0, 8, v16, v17);
      v19 = (unsigned int)v93;
LABEL_15:
      *(_QWORD *)&v92[8 * v19] = v18;
      v20 = *(_QWORD *)(a1 + 296);
      v81 = 257;
      v21 = *(__int64 ***)(a1 + 304);
      LODWORD(v93) = v93 + 1;
      v22 = sub_15A0680(v20, 0, 0);
      v11 = v22;
      if ( v21 == *(__int64 ***)v22 )
        goto LABEL_18;
      if ( *(_BYTE *)(v22 + 16) <= 0x10u )
      {
        v11 = sub_15A46C0(46, (__int64 ***)v22, v21, 0);
LABEL_18:
        v23 = (unsigned int)v93;
        if ( (unsigned int)v93 >= HIDWORD(v93) )
          goto LABEL_52;
        goto LABEL_19;
      }
      v83 = 257;
      v49 = sub_15FDBD0(46, v22, (__int64)v21, (__int64)v82, 0);
      v50 = v49;
      if ( v85 )
      {
        v73 = v49;
        v68 = (__int64 *)v86;
        sub_157E9D0(v85 + 40, v49);
        v50 = v73;
        v51 = *(_QWORD *)(v73 + 24);
        v52 = *v68;
        *(_QWORD *)(v73 + 32) = v68;
        v52 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v73 + 24) = v52 | v51 & 7;
        *(_QWORD *)(v52 + 8) = v73 + 24;
        *v68 = *v68 & 7 | (v73 + 24);
      }
      v74 = v50;
      sub_164B780(v50, v80);
      v11 = v74;
      if ( !v84 )
        goto LABEL_18;
      v79 = v84;
      sub_1623A60((__int64)&v79, (__int64)v84, 2);
      v11 = v74;
      v53 = *(_QWORD *)(v74 + 48);
      v54 = v74 + 48;
      if ( v53 )
      {
        sub_161E7C0(v74 + 48, v53);
        v11 = v74;
        v54 = v74 + 48;
      }
      v55 = v79;
      *(_QWORD *)(v11 + 48) = v79;
      if ( !v55 )
        goto LABEL_18;
      v75 = v11;
      sub_1623210((__int64)&v79, v55, v54);
      v11 = v75;
      v23 = (unsigned int)v93;
      if ( (unsigned int)v93 >= HIDWORD(v93) )
      {
LABEL_52:
        v76 = v11;
        sub_16CD150((__int64)&v92, v94, 0, 8, v10, v11);
        v23 = (unsigned int)v93;
        v11 = v76;
      }
LABEL_19:
      ++v14;
      *(_QWORD *)&v92[8 * v23] = v11;
      LODWORD(v93) = v93 + 1;
    }
    while ( v14 != a4 );
  }
LABEL_32:
  v34 = 2 * a4;
  v35 = sub_17F58C0(a1, v34, a2, *(_QWORD *)(a1 + 304), "sancov_pcs");
  v36 = (unsigned int)v93;
  v37 = (__int64)v35;
  *((_QWORD *)&v38 + 1) = v92;
  *(_QWORD *)&v38 = sub_1645D80(*(__int64 **)(a1 + 304), v34);
  v40 = sub_159DFD0(v38, v36, v39);
  sub_15E5440(v37, v40);
  v41 = v84;
  *(_BYTE *)(v37 + 80) |= 1u;
  if ( v41 )
    sub_161E7C0((__int64)&v84, (__int64)v41);
  if ( v92 != v94 )
    _libc_free((unsigned __int64)v92);
  return v37;
}
