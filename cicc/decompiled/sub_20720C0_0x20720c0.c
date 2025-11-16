// Function: sub_20720C0
// Address: 0x20720c0
//
void __fastcall sub_20720C0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 *v8; // r14
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r11
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r9
  int v17; // r8d
  _BYTE *v18; // rax
  __int64 v19; // r9
  _BYTE *v20; // rdx
  __int64 *v21; // rsi
  __int64 v22; // r14
  __int64 *v23; // rax
  __int64 v24; // r9
  __int64 *v25; // r10
  __int64 *v26; // r15
  int v27; // edx
  int v28; // r13d
  __int64 v29; // r12
  int v30; // ebx
  char v31; // r13
  __int64 v32; // rax
  _BYTE *v33; // rax
  __int64 v34; // rdx
  __int64 *v35; // rcx
  unsigned __int8 *v36; // rdx
  unsigned int v37; // ecx
  __int64 v38; // r8
  _QWORD *v39; // rdi
  __int64 *v40; // rax
  __int64 *v41; // r13
  __int64 v42; // r14
  __int64 v43; // r15
  __int64 v44; // rax
  int v45; // edx
  __int64 v46; // r9
  __int64 v47; // r10
  const void ***v48; // rcx
  __int64 v49; // rax
  int v50; // r8d
  bool v51; // zf
  __int64 v52; // rsi
  __int64 *v53; // r13
  int v54; // edx
  int v55; // r14d
  __int64 *v56; // rax
  unsigned __int8 *v57; // rdi
  _QWORD *v58; // rdi
  _QWORD *v59; // r13
  int v60; // edx
  int v61; // r14d
  __int64 *v62; // rax
  __int128 v63; // [rsp-10h] [rbp-160h]
  __int64 v64; // [rsp+8h] [rbp-148h]
  char v65; // [rsp+18h] [rbp-138h]
  int v66; // [rsp+18h] [rbp-138h]
  int v67; // [rsp+20h] [rbp-130h]
  __int64 v68; // [rsp+20h] [rbp-130h]
  int v69; // [rsp+20h] [rbp-130h]
  __int64 v70; // [rsp+30h] [rbp-120h]
  int v71; // [rsp+30h] [rbp-120h]
  __int64 v72; // [rsp+30h] [rbp-120h]
  __int64 *v73; // [rsp+30h] [rbp-120h]
  const void ***v74; // [rsp+30h] [rbp-120h]
  int v75; // [rsp+30h] [rbp-120h]
  __int64 v76; // [rsp+38h] [rbp-118h]
  __int64 v77; // [rsp+38h] [rbp-118h]
  __int64 v78; // [rsp+38h] [rbp-118h]
  __int64 v79; // [rsp+38h] [rbp-118h]
  __int64 v80; // [rsp+38h] [rbp-118h]
  __int64 v81; // [rsp+68h] [rbp-E8h] BYREF
  __int64 v82; // [rsp+70h] [rbp-E0h] BYREF
  int v83; // [rsp+78h] [rbp-D8h]
  unsigned __int8 *v84; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v85; // [rsp+88h] [rbp-C8h]
  _BYTE v86[64]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE *v87; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v88; // [rsp+D8h] [rbp-78h]
  _BYTE v89[112]; // [rsp+E0h] [rbp-70h] BYREF

  v5 = a2;
  v6 = a1;
  v7 = *(_QWORD *)a2;
  if ( *(_BYTE *)(a2 + 16) == 5 )
  {
    v8 = *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v9 = *v8;
    v10 = sub_1594710(a2);
    v67 = sub_20C7BE0(v9, v10, v10 + 4 * v11, 0);
  }
  else
  {
    v8 = *(__int64 **)(a2 - 24);
    v67 = sub_20C7BE0(*v8, *(_QWORD *)(a2 + 56), *(_QWORD *)(a2 + 56) + 4LL * *(unsigned int *)(a2 + 64), 0);
  }
  v65 = *((_BYTE *)v8 + 16);
  v12 = *(_QWORD *)(a1 + 552);
  v13 = *(_QWORD *)(v12 + 16);
  v14 = *(_QWORD *)(v12 + 32);
  v84 = v86;
  v85 = 0x400000000LL;
  v70 = v13;
  v15 = sub_1E0A0C0(v14);
  sub_20C7CE0(v70, v15, v7, &v84, 0, 0);
  v17 = v85;
  if ( (_DWORD)v85 )
  {
    v18 = v89;
    v88 = 0x400000000LL;
    v19 = (unsigned int)v85;
    v87 = v89;
    if ( (unsigned int)v85 > 4 )
    {
      v75 = v85;
      v80 = (unsigned int)v85;
      sub_16CD150((__int64)&v87, v89, (unsigned int)v85, 16, v85, v85);
      v18 = v87;
      v17 = v75;
      v19 = v80;
    }
    LODWORD(v88) = v17;
    v20 = &v18[16 * v19];
    do
    {
      if ( v18 )
      {
        *(_QWORD *)v18 = 0;
        *((_DWORD *)v18 + 2) = 0;
      }
      v18 += 16;
    }
    while ( v20 != v18 );
    v21 = v8;
    v22 = 0;
    v71 = v17;
    v76 = v19;
    v23 = sub_20685E0(v6, v21, a3, a4, a5);
    v24 = v76;
    v25 = &v82;
    v26 = v23;
    v28 = v67 + v27;
    if ( v71 )
    {
      v68 = v5;
      v29 = v6;
      v30 = v28;
      v31 = v65;
      do
      {
        v34 = (unsigned int)(v30 + v22);
        v35 = v26;
        if ( v31 == 9 )
        {
          v36 = (unsigned __int8 *)(v26[5] + 16 * v34);
          v37 = *v36;
          v38 = *((_QWORD *)v36 + 1);
          v72 = v24;
          v39 = *(_QWORD **)(v29 + 552);
          v77 = (__int64)v25;
          v82 = 0;
          v83 = 0;
          v40 = sub_1D2B300(v39, 0x30u, (__int64)v25, v37, v38, v24);
          v25 = (__int64 *)v77;
          v24 = v72;
          v35 = v40;
          if ( v82 )
          {
            v64 = v72;
            v66 = v34;
            v73 = v40;
            sub_161E7C0(v77, v82);
            v24 = v64;
            LODWORD(v34) = v66;
            v35 = v73;
            v25 = (__int64 *)v77;
          }
        }
        v32 = v22++;
        v33 = &v87[16 * v32];
        *(_QWORD *)v33 = v35;
        *((_DWORD *)v33 + 2) = v34;
      }
      while ( v24 != v22 );
      v6 = v29;
      v5 = v68;
    }
    v41 = *(__int64 **)(v6 + 552);
    v78 = (__int64)v25;
    v42 = (__int64)v87;
    v43 = (unsigned int)v88;
    v44 = sub_1D25C30((__int64)v41, v84, (unsigned int)v85);
    v47 = v78;
    v82 = 0;
    v48 = (const void ***)v44;
    v49 = *(_QWORD *)v6;
    v50 = v45;
    v51 = *(_QWORD *)v6 == 0;
    v83 = *(_DWORD *)(v6 + 536);
    if ( !v51 && v78 != v49 + 48 )
    {
      v52 = *(_QWORD *)(v49 + 48);
      v82 = v52;
      if ( v52 )
      {
        v69 = v45;
        v74 = v48;
        sub_1623A60(v78, v52, 2);
        v50 = v69;
        v48 = v74;
        v47 = v78;
      }
    }
    *((_QWORD *)&v63 + 1) = v43;
    *(_QWORD *)&v63 = v42;
    v79 = v47;
    v81 = v5;
    v53 = sub_1D36D80(v41, 51, v47, v48, v50, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5, v46, v63);
    v55 = v54;
    v56 = sub_205F5C0(v6 + 8, &v81);
    v56[1] = (__int64)v53;
    *((_DWORD *)v56 + 4) = v55;
    if ( v82 )
      sub_161E7C0(v79, v82);
    if ( v87 != v89 )
      _libc_free((unsigned __int64)v87);
    v57 = v84;
    if ( v84 != v86 )
LABEL_26:
      _libc_free((unsigned __int64)v57);
  }
  else
  {
    v58 = *(_QWORD **)(v6 + 552);
    v87 = 0;
    LODWORD(v88) = 0;
    v59 = sub_1D2B300(v58, 0x30u, (__int64)&v87, 1u, 0, v16);
    v61 = v60;
    if ( v87 )
      sub_161E7C0((__int64)&v87, (__int64)v87);
    v87 = (_BYTE *)a2;
    v62 = sub_205F5C0(v6 + 8, (__int64 *)&v87);
    v57 = v84;
    v62[1] = (__int64)v59;
    *((_DWORD *)v62 + 4) = v61;
    if ( v57 != v86 )
      goto LABEL_26;
  }
}
