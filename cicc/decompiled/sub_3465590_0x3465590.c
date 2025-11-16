// Function: sub_3465590
// Address: 0x3465590
//
unsigned __int8 *__fastcall sub_3465590(
        __m128i a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        __int64 a7,
        unsigned __int16 a8,
        __int64 a9,
        __int64 a10,
        char a11)
{
  __int64 v12; // r15
  __int64 v13; // rax
  unsigned __int16 v14; // bx
  __int64 v15; // rax
  unsigned __int16 v16; // dx
  bool v17; // al
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rdx
  char v21; // al
  unsigned int v22; // eax
  __int64 v23; // rcx
  unsigned int v24; // r8d
  __int16 v25; // ax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // rsi
  __int64 v30; // rdx
  unsigned __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // r9
  unsigned __int8 *v35; // r12
  unsigned int v36; // edx
  unsigned __int64 v37; // r13
  __int64 v39; // rax
  unsigned int v40; // edx
  unsigned __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdx
  int v44; // r9d
  unsigned __int8 *v45; // rax
  unsigned int v46; // edx
  __int64 v47; // r13
  unsigned int v48; // edx
  unsigned __int8 *v49; // r12
  unsigned __int64 v50; // r10
  unsigned __int16 v51; // dx
  unsigned __int64 v52; // r13
  __int64 v53; // rax
  unsigned __int64 v54; // rsi
  __int128 v55; // rax
  __int64 v56; // r9
  unsigned int v57; // edx
  bool v58; // al
  __int64 v59; // rcx
  __int64 v60; // r8
  unsigned __int16 v61; // ax
  __int64 v62; // rdx
  __int64 v63; // r8
  __int64 v64; // rax
  unsigned int v65; // edx
  __int128 v66; // [rsp-30h] [rbp-170h]
  __int128 v67; // [rsp-10h] [rbp-150h]
  unsigned int v68; // [rsp+0h] [rbp-140h]
  __int64 v69; // [rsp+8h] [rbp-138h]
  unsigned int v70; // [rsp+8h] [rbp-138h]
  unsigned int v71; // [rsp+8h] [rbp-138h]
  unsigned __int16 v72; // [rsp+10h] [rbp-130h]
  __int64 v73; // [rsp+10h] [rbp-130h]
  __int64 v74; // [rsp+10h] [rbp-130h]
  unsigned __int64 v75; // [rsp+18h] [rbp-128h]
  __int128 v76; // [rsp+20h] [rbp-120h]
  unsigned int v77; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v78; // [rsp+98h] [rbp-A8h]
  unsigned __int16 v79; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v80; // [rsp+A8h] [rbp-98h]
  __int64 v81; // [rsp+B0h] [rbp-90h]
  __int64 v82; // [rsp+B8h] [rbp-88h]
  __int64 v83; // [rsp+C0h] [rbp-80h]
  __int64 v84; // [rsp+C8h] [rbp-78h]
  __int64 v85; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v86; // [rsp+E8h] [rbp-58h]
  unsigned __int64 v87; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v88; // [rsp+F8h] [rbp-48h]
  unsigned __int64 v89; // [rsp+100h] [rbp-40h] BYREF
  __int64 v90; // [rsp+108h] [rbp-38h]

  *(_QWORD *)&v76 = a3;
  v12 = a10;
  *((_QWORD *)&v76 + 1) = a4;
  v13 = *(_QWORD *)(a3 + 48) + 16LL * (unsigned int)a4;
  v14 = *(_WORD *)v13;
  v78 = *(_QWORD *)(v13 + 8);
  v15 = *(_QWORD *)(a5 + 48) + 16LL * a6;
  LOWORD(v77) = v14;
  v16 = *(_WORD *)v15;
  v80 = *(_QWORD *)(v15 + 8);
  v79 = v16;
  if ( a11 )
  {
    if ( a8 )
    {
      if ( (unsigned __int16)(a8 - 176) > 0x34u )
      {
LABEL_4:
        if ( v16 )
        {
          if ( v16 == 1 || (unsigned __int16)(v16 - 504) <= 7u )
            goto LABEL_70;
          v64 = 16LL * (v16 - 1);
          v20 = *(_QWORD *)&byte_444C4A0[v64];
          v21 = byte_444C4A0[v64 + 8];
        }
        else
        {
          v70 = a6;
          v73 = a5;
          v18 = sub_3007260((__int64)&v79);
          a6 = v70;
          a5 = v73;
          v83 = v18;
          v84 = v19;
          v20 = v18;
          v21 = v84;
        }
        v71 = a6;
        v74 = a5;
        v89 = v20;
        LOBYTE(v90) = v21;
        v22 = sub_CA1930(&v89);
        v23 = v74;
        v24 = v71;
        switch ( v22 )
        {
          case 1u:
            v25 = 2;
            break;
          case 2u:
            v25 = 3;
            break;
          case 4u:
            v25 = 4;
            break;
          case 8u:
            v25 = 5;
            break;
          case 0x10u:
            v25 = 6;
            break;
          case 0x20u:
            v25 = 7;
            break;
          case 0x40u:
            v25 = 8;
            break;
          case 0x80u:
            v25 = 9;
            break;
          default:
            v25 = sub_3007020(*(_QWORD **)(v12 + 64), v22);
            v23 = v74;
            v24 = v71;
            goto LABEL_37;
        }
        v26 = 0;
LABEL_37:
        LOWORD(v85) = v25;
        v86 = v26;
        sub_33FB890(v12, v85, v26, v23, v24, a1);
        v75 = v41;
        if ( (_WORD)v85 )
        {
          if ( (_WORD)v85 == 1 || (unsigned __int16)(v85 - 504) <= 7u )
            goto LABEL_70;
          v43 = 16LL * ((unsigned __int16)v85 - 1);
          v42 = *(_QWORD *)&byte_444C4A0[v43];
          LOBYTE(v43) = byte_444C4A0[v43 + 8];
        }
        else
        {
          v42 = sub_3007260((__int64)&v85);
          v81 = v42;
          v82 = v43;
        }
        v89 = v42;
        LOBYTE(v90) = v43;
        if ( (unsigned __int64)sub_CA1930(&v89) <= 0x1F )
        {
          sub_33FAF80(v12, 214, a7, 7, 0, v44, a1);
          v44 = v75;
          LOWORD(v85) = 7;
          v86 = 0;
          v75 = v65 | v75 & 0xFFFFFFFF00000000LL;
        }
        v45 = sub_33FAF80(v12, 200, a7, (unsigned int)v85, v86, v44, a1);
        v47 = v46;
        v49 = sub_33FB310(v12, (__int64)v45, v46, a7, v77, v78, a1);
        v50 = v48 | v47 & 0xFFFFFFFF00000000LL;
        v51 = a8;
        v52 = v50;
        if ( a8 )
        {
          if ( (unsigned __int16)(a8 - 17) <= 0xD3u )
          {
            v51 = word_4456580[a8 - 1];
            v53 = 0;
            goto LABEL_44;
          }
        }
        else
        {
          v58 = sub_30070B0((__int64)&a8);
          v51 = 0;
          if ( v58 )
          {
            v61 = sub_3009970((__int64)&a8, v75, 0, v59, v60);
            v63 = v62;
            v51 = v61;
            v53 = v63;
            goto LABEL_44;
          }
        }
        v53 = a9;
LABEL_44:
        LOWORD(v89) = v51;
        v90 = v53;
        if ( !v51 )
        {
          v54 = sub_3007260((__int64)&v89);
LABEL_46:
          *(_QWORD *)&v55 = sub_3400BD0(v12, v54 >> 3, a7, v77, v78, 0, a1, 0);
          *((_QWORD *)&v66 + 1) = v52;
          *(_QWORD *)&v66 = v49;
          v35 = sub_3406EB0((_QWORD *)v12, 0x3Au, a7, v77, v78, v56, v66, v55);
          v37 = v57 | v52 & 0xFFFFFFFF00000000LL;
          goto LABEL_29;
        }
        if ( v51 != 1 && (unsigned __int16)(v51 - 504) > 7u )
        {
          v54 = *(_QWORD *)&byte_444C4A0[16 * v51 - 16];
          goto LABEL_46;
        }
LABEL_70:
        BUG();
      }
    }
    else
    {
      v68 = a6;
      v69 = a5;
      v72 = v16;
      v17 = sub_3007100((__int64)&a8);
      v16 = v72;
      a5 = v69;
      a6 = v68;
      if ( !v17 )
        goto LABEL_4;
    }
    sub_C64ED0("Cannot currently handle compressed memory with scalable vectors", 1u);
  }
  if ( a8 )
  {
    if ( (unsigned __int16)(a8 - 176) > 0x34u )
    {
      if ( a8 == 1 || (unsigned __int16)(a8 - 504) <= 7u )
        goto LABEL_70;
      v27 = *(_QWORD *)&byte_444C4A0[16 * a8 - 16];
      LOBYTE(v28) = byte_444C4A0[16 * a8 - 8];
      goto LABEL_33;
    }
    v29 = *(_QWORD *)&byte_444C4A0[16 * a8 - 16];
  }
  else
  {
    if ( !sub_3007100((__int64)&a8) )
    {
      v27 = sub_3007260((__int64)&a8);
      v89 = v27;
      v90 = v28;
LABEL_33:
      LOBYTE(v88) = v28;
      v87 = (unsigned __int64)(v27 + 7) >> 3;
      v39 = sub_CA1930(&v87);
      v35 = sub_3400BD0(v12, v39, a7, v77, v78, 0, a1, 0);
      v37 = v40;
      goto LABEL_29;
    }
    v87 = sub_3007260((__int64)&a8);
    v29 = v87;
    v88 = v30;
  }
  v31 = (v29 + 7) >> 3;
  if ( v14 )
  {
    if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
      goto LABEL_70;
    v32 = *(_QWORD *)&byte_444C4A0[16 * v14 - 16];
  }
  else
  {
    v32 = sub_3007260((__int64)&v77);
    v85 = v32;
    v86 = v33;
  }
  LODWORD(v90) = v32;
  if ( (unsigned int)v32 > 0x40 )
    sub_C43690((__int64)&v89, v31, 0);
  else
    v89 = v31;
  v35 = sub_3401900(v12, a7, v77, v78, (__int64)&v89, 1, a1);
  v37 = v36;
  if ( (unsigned int)v90 > 0x40 && v89 )
    j_j___libc_free_0_0(v89);
LABEL_29:
  *((_QWORD *)&v67 + 1) = v37;
  *(_QWORD *)&v67 = v35;
  return sub_3406EB0((_QWORD *)v12, 0x38u, a7, v77, v78, v34, v76, v67);
}
