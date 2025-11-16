// Function: sub_262AEE0
// Address: 0x262aee0
//
void __fastcall sub_262AEE0(__int64 a1, __int64 *a2, __int64 a3, __int64 **a4, __int64 a5)
{
  __int64 *v5; // rax
  __int64 v6; // r13
  __int64 **v7; // r14
  __int64 v8; // r11
  unsigned __int64 v9; // rbx
  unsigned __int8 v10; // al
  unsigned __int8 v11; // cl
  unsigned __int8 v12; // al
  __int64 v13; // rbx
  __int64 v14; // rbx
  unsigned int v15; // r8d
  _QWORD *v16; // rax
  __int64 *v17; // rdi
  __int64 *v18; // rax
  unsigned __int64 *v19; // rax
  unsigned __int64 *v20; // rsi
  unsigned __int64 v21; // rdx
  char v22; // r15
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // rbx
  __int64 *v27; // r12
  __int64 v28; // r15
  __int16 v29; // cx
  __int64 **v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  unsigned __int64 *v33; // rax
  __int64 *v34; // r13
  __int64 v35; // r12
  __int64 **v36; // rax
  __int64 v37; // rax
  _QWORD *v38; // r14
  __int64 v39; // rbx
  unsigned __int8 *v40; // r13
  __int64 v41; // r9
  int v42; // ebx
  __int64 v43; // rax
  __int64 *v44; // r14
  unsigned __int8 *v45; // r13
  __int64 v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rdi
  __int64 v49; // rax
  __int64 v50; // r8
  __int64 v51; // r9
  unsigned __int8 *v52; // r15
  unsigned __int8 v53; // al
  __int64 **v54; // rcx
  int v55; // eax
  unsigned int v56; // r9d
  __int64 *v57; // rsi
  int v58; // r8d
  __int64 **v59; // r10
  int v60; // r8d
  __int64 v61; // r9
  __int64 *v62; // rdx
  __int64 v63; // [rsp+0h] [rbp-100h]
  __int64 v67; // [rsp+18h] [rbp-E8h]
  __int64 v68; // [rsp+18h] [rbp-E8h]
  unsigned __int8 *v69; // [rsp+20h] [rbp-E0h]
  __int64 **v71; // [rsp+30h] [rbp-D0h]
  __int64 v73; // [rsp+40h] [rbp-C0h]
  __int64 v74; // [rsp+40h] [rbp-C0h]
  __int64 v75; // [rsp+40h] [rbp-C0h]
  int v76; // [rsp+40h] [rbp-C0h]
  __int64 v77; // [rsp+40h] [rbp-C0h]
  unsigned __int8 v78; // [rsp+48h] [rbp-B8h]
  __int64 v79; // [rsp+48h] [rbp-B8h]
  __int64 v80[2]; // [rsp+50h] [rbp-B0h] BYREF
  __int64 *v81; // [rsp+60h] [rbp-A0h] BYREF
  unsigned __int64 *v82; // [rsp+68h] [rbp-98h]
  unsigned __int64 *v83; // [rsp+70h] [rbp-90h]
  __int64 v84; // [rsp+80h] [rbp-80h] BYREF
  __int64 v85; // [rsp+88h] [rbp-78h]
  __int64 v86; // [rsp+90h] [rbp-70h]
  unsigned int v87; // [rsp+98h] [rbp-68h]
  unsigned __int64 v88; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v89; // [rsp+A8h] [rbp-58h]
  unsigned __int64 v90; // [rsp+B0h] [rbp-50h]
  unsigned int v91; // [rsp+B8h] [rbp-48h]
  __int16 v92; // [rsp+C0h] [rbp-40h]

  v5 = *(__int64 **)a1;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v71 = &a4[a5];
  if ( v71 != a4 )
  {
    v6 = (__int64)(v5 + 39);
    v7 = a4;
    v8 = 0;
    v78 = 0;
    v9 = 0;
    while ( 1 )
    {
      v27 = *v7;
      v28 = **v7;
      v29 = (*(_WORD *)(v28 + 34) >> 1) & 0x3F;
      if ( v29 )
      {
        v11 = v29 - 1;
      }
      else
      {
        v73 = v8;
        v10 = sub_AE5020(v6, *(_QWORD *)(v28 + 24));
        v8 = v73;
        v11 = v10;
      }
      v12 = v78;
      v13 = v9 + v8 - 1;
      if ( v78 < v11 )
        v12 = v11;
      v78 = v12;
      v14 = -(1LL << v11) & ((1LL << v11) + v13);
      if ( v87 )
      {
        v15 = (v87 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v16 = (_QWORD *)(v85 + 16LL * v15);
        v17 = (__int64 *)*v16;
        if ( v27 == (__int64 *)*v16 )
        {
LABEL_8:
          v18 = v16 + 1;
          goto LABEL_9;
        }
        v76 = 1;
        v54 = 0;
        while ( v17 != (__int64 *)-4096LL )
        {
          if ( !v54 && v17 == (__int64 *)-8192LL )
            v54 = (__int64 **)v16;
          v15 = (v87 - 1) & (v76 + v15);
          v16 = (_QWORD *)(v85 + 16LL * v15);
          v17 = (__int64 *)*v16;
          if ( v27 == (__int64 *)*v16 )
            goto LABEL_8;
          ++v76;
        }
        if ( !v54 )
          v54 = (__int64 **)v16;
        ++v84;
        v55 = v86 + 1;
        if ( 4 * ((int)v86 + 1) < 3 * v87 )
        {
          if ( v87 - HIDWORD(v86) - v55 > v87 >> 3 )
            goto LABEL_52;
          v63 = v8;
          sub_261CD60((__int64)&v84, v87);
          if ( !v87 )
          {
LABEL_83:
            LODWORD(v86) = v86 + 1;
            BUG();
          }
          v59 = 0;
          v8 = v63;
          v60 = 1;
          v55 = v86 + 1;
          v61 = (v87 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v54 = (__int64 **)(v85 + 16 * v61);
          v62 = *v54;
          if ( v27 == *v54 )
            goto LABEL_52;
          while ( v62 != (__int64 *)-4096LL )
          {
            if ( v62 == (__int64 *)-8192LL && !v59 )
              v59 = v54;
            LODWORD(v61) = (v87 - 1) & (v60 + v61);
            v54 = (__int64 **)(v85 + 16LL * (unsigned int)v61);
            v62 = *v54;
            if ( v27 == *v54 )
              goto LABEL_52;
            ++v60;
          }
          goto LABEL_61;
        }
      }
      else
      {
        ++v84;
      }
      v77 = v8;
      sub_261CD60((__int64)&v84, 2 * v87);
      if ( !v87 )
        goto LABEL_83;
      v8 = v77;
      v56 = (v87 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v55 = v86 + 1;
      v54 = (__int64 **)(v85 + 16LL * v56);
      v57 = *v54;
      if ( v27 == *v54 )
        goto LABEL_52;
      v58 = 1;
      v59 = 0;
      while ( v57 != (__int64 *)-4096LL )
      {
        if ( !v59 && v57 == (__int64 *)-8192LL )
          v59 = v54;
        v56 = (v87 - 1) & (v58 + v56);
        v54 = (__int64 **)(v85 + 16LL * v56);
        v57 = *v54;
        if ( v27 == *v54 )
          goto LABEL_52;
        ++v58;
      }
LABEL_61:
      if ( v59 )
        v54 = v59;
LABEL_52:
      LODWORD(v86) = v55;
      if ( *v54 != (__int64 *)-4096LL )
        --HIDWORD(v86);
      *v54 = v27;
      v18 = (__int64 *)(v54 + 1);
      v54[1] = 0;
LABEL_9:
      *v18 = v14;
      if ( !v14 )
      {
        v19 = v82;
        v20 = v83;
        goto LABEL_11;
      }
      v30 = (__int64 **)sub_BCD420(*(__int64 **)(a1 + 64), v14 - v8);
      v31 = sub_AC9350(v30);
      v20 = v83;
      v88 = v31;
      v32 = v31;
      v33 = v82;
      if ( v82 == v83 )
      {
        sub_262AD50((__int64)&v81, v83, &v88);
        v19 = v82;
        v20 = v83;
LABEL_11:
        v21 = *(_QWORD *)(v28 - 32);
        v88 = v21;
        if ( v20 != v19 )
          goto LABEL_12;
        goto LABEL_24;
      }
      if ( v82 )
      {
        *v82 = v32;
        v33 = v82;
        v20 = v83;
      }
      v19 = v33 + 1;
      v82 = v19;
      v21 = *(_QWORD *)(v28 - 32);
      v88 = v21;
      if ( v20 != v19 )
      {
LABEL_12:
        if ( v19 )
        {
          *v19 = v21;
          v19 = v82;
        }
        v82 = v19 + 1;
        goto LABEL_15;
      }
LABEL_24:
      sub_262AD50((__int64)&v81, v20, &v88);
LABEL_15:
      v74 = *(_QWORD *)(v28 + 24);
      v22 = sub_AE5020(v6, v74);
      v23 = sub_9208B0(v6, v74);
      v89 = v24;
      v88 = (((unsigned __int64)(v23 + 7) >> 3) + (1LL << v22) - 1) >> v22 << v22;
      v25 = sub_CA1930(&v88);
      v8 = v25 + v14;
      v26 = (((((((((v25 - 1) | ((unsigned __int64)(v25 - 1) >> 1)) >> 2)
                | (v25 - 1)
                | ((unsigned __int64)(v25 - 1) >> 1)) >> 4)
              | (((v25 - 1) | ((unsigned __int64)(v25 - 1) >> 1)) >> 2)
              | (v25 - 1)
              | ((unsigned __int64)(v25 - 1) >> 1)) >> 8)
            | (((((v25 - 1) | ((unsigned __int64)(v25 - 1) >> 1)) >> 2) | (v25 - 1) | ((unsigned __int64)(v25 - 1) >> 1)) >> 4)
            | (((v25 - 1) | ((unsigned __int64)(v25 - 1) >> 1)) >> 2)
            | (v25 - 1)
            | ((unsigned __int64)(v25 - 1) >> 1)) >> 16)
          | (((((((v25 - 1) | ((unsigned __int64)(v25 - 1) >> 1)) >> 2) | (v25 - 1) | ((unsigned __int64)(v25 - 1) >> 1)) >> 4)
            | (((v25 - 1) | ((unsigned __int64)(v25 - 1) >> 1)) >> 2)
            | (v25 - 1)
            | ((unsigned __int64)(v25 - 1) >> 1)) >> 8)
          | (((((v25 - 1) | ((unsigned __int64)(v25 - 1) >> 1)) >> 2) | (v25 - 1) | ((unsigned __int64)(v25 - 1) >> 1)) >> 4)
          | (((v25 - 1) | ((unsigned __int64)(v25 - 1) >> 1)) >> 2)
          | (v25 - 1)
          | ((unsigned __int64)(v25 - 1) >> 1);
      v9 = 1 - v25 + (HIDWORD(v26) | v26);
      if ( v9 > 0x20 )
        v9 = 32 * ((v25 != 0) + ((v25 - (unsigned __int64)(v25 != 0)) >> 5)) - v25;
      if ( v71 == ++v7 )
      {
        v34 = v81;
        v5 = *(__int64 **)a1;
        v35 = ((char *)v82 - (char *)v81) >> 3;
        goto LABEL_26;
      }
    }
  }
  v78 = 0;
  v35 = 0;
  v34 = 0;
LABEL_26:
  v36 = (__int64 **)sub_AC3380(*v5, (__int64)v34, v35, 0);
  v37 = sub_AD24A0(v36, v34, v35);
  v38 = *(_QWORD **)(v37 + 8);
  v75 = v37;
  v39 = v37;
  v92 = 257;
  BYTE4(v80[0]) = 0;
  v40 = (unsigned __int8 *)sub_BD2C40(88, unk_3F0FAE8);
  if ( v40 )
    sub_B30000((__int64)v40, *(_QWORD *)a1, v38, 1, 8, v39, (__int64)&v88, 0, 0, v80[0], 0);
  sub_B2F770((__int64)v40, v78);
  v79 = *(_QWORD *)(v75 + 8);
  sub_26297F0(a1, a2, a3, v40, (__int64)&v84, v41);
  if ( a5 )
  {
    v42 = 0;
    v43 = 0;
    v44 = (__int64 *)a1;
    v69 = v40;
    do
    {
      v45 = (unsigned __int8 *)*a4[v43];
      v46 = sub_ACD640(v44[11], 0, 0);
      v47 = v44[11];
      v80[0] = v46;
      v80[1] = sub_ACD640(v47, (unsigned int)(2 * v42), 0);
      v48 = *(_QWORD *)(v75 + 8);
      LOBYTE(v92) = 0;
      v49 = sub_AD9FD0(v48, v69, v80, 2, 3u, (__int64)&v88, 0);
      v50 = v49;
      if ( (_BYTE)v92 )
      {
        LOBYTE(v92) = 0;
        if ( v91 > 0x40 && v90 )
        {
          v67 = v49;
          j_j___libc_free_0_0(v90);
          v50 = v67;
        }
        if ( (unsigned int)v89 > 0x40 && v88 )
        {
          v68 = v50;
          j_j___libc_free_0_0(v88);
          v50 = v68;
        }
      }
      v51 = *v44;
      v92 = 257;
      v52 = (unsigned __int8 *)sub_B30500(
                                 *(_QWORD **)(*(_QWORD *)(v79 + 16) + 8LL * (unsigned int)(2 * v42)),
                                 0,
                                 v45[32] & 0xF,
                                 (__int64)&v88,
                                 v50,
                                 v51);
      v53 = v45[32] & 0x30 | v52[32] & 0xCF;
      v52[32] = v53;
      if ( (v53 & 0xFu) - 7 <= 1 || (v53 & 0x30) != 0 && (v53 & 0xF) != 9 )
        v52[33] |= 0x40u;
      sub_BD6B90(v52, v45);
      sub_BD84D0((__int64)v45, (__int64)v52);
      sub_B30290((__int64)v45);
      v43 = (unsigned int)++v42;
    }
    while ( a5 != v42 );
  }
  sub_C7D6A0(v85, 16LL * v87, 8);
  if ( v81 )
    j_j___libc_free_0((unsigned __int64)v81);
}
