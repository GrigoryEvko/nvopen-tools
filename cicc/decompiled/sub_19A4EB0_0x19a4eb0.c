// Function: sub_19A4EB0
// Address: 0x19a4eb0
//
void __fastcall sub_19A4EB0(__int64 a1, __m128i a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, int a7, int a8)
{
  _BYTE *v8; // r12
  __int64 v9; // rbx
  unsigned __int64 v10; // rdx
  const void *v11; // r15
  size_t v12; // r14
  int v13; // r13d
  _BYTE *v14; // rdi
  __int64 *v15; // r9
  int v16; // eax
  __int64 *v17; // r8
  char v18; // dl
  __int64 v19; // r15
  int v20; // edx
  __int64 v21; // rax
  const void *v22; // r8
  signed __int64 v23; // r14
  __int64 v24; // r13
  __int64 v25; // rsi
  __int64 *v26; // rdi
  __int64 *v27; // rax
  __int64 *v28; // rcx
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 v31; // r13
  unsigned __int8 v32; // al
  _QWORD *v33; // r14
  _QWORD *v34; // r13
  unsigned __int8 v35; // al
  __int64 v36; // r12
  __int64 *v37; // rdx
  __int64 v38; // rsi
  unsigned __int64 v39; // rax
  __int64 v40; // rcx
  int v41; // eax
  __int64 v42; // rax
  __int64 v43; // rax
  _QWORD *v44; // r9
  __int64 v45; // r8
  __int64 v46; // rdx
  unsigned int v47; // edx
  __int64 v48; // rax
  bool v49; // zf
  __int64 v50; // rax
  __int64 v51; // r13
  __int64 v52; // r13
  bool v53; // al
  __int64 v54; // r8
  __int64 v55; // rsi
  unsigned __int64 v56; // r14
  unsigned __int64 v57; // rax
  int v58; // r9d
  __int64 v59; // r8
  __int64 v60; // r13
  __int64 v61; // r14
  __int64 v62; // rsi
  __int64 *v63; // rax
  __int64 *v64; // rcx
  __int64 *v65; // r14
  _BYTE *v66; // rax
  __int64 v67; // r12
  __int64 *v68; // rbx
  _BYTE *v69; // r15
  __int64 v70; // rsi
  int v71; // eax
  _QWORD *v72; // rcx
  __int64 v73; // rax
  __int64 v74; // [rsp+8h] [rbp-248h]
  __int64 v75; // [rsp+8h] [rbp-248h]
  __int64 v76; // [rsp+10h] [rbp-240h]
  _QWORD *v77; // [rsp+10h] [rbp-240h]
  _QWORD *v78; // [rsp+10h] [rbp-240h]
  __int64 v79; // [rsp+10h] [rbp-240h]
  __int64 v80; // [rsp+10h] [rbp-240h]
  __int64 v81; // [rsp+18h] [rbp-238h]
  __int64 v82; // [rsp+18h] [rbp-238h]
  __int64 v83; // [rsp+18h] [rbp-238h]
  const void *v84; // [rsp+20h] [rbp-230h]
  _BYTE *v85; // [rsp+20h] [rbp-230h]
  unsigned int v86; // [rsp+20h] [rbp-230h]
  __int64 v87; // [rsp+20h] [rbp-230h]
  __int64 v88; // [rsp+30h] [rbp-220h] BYREF
  __int64 v89; // [rsp+38h] [rbp-218h] BYREF
  _BYTE *v90; // [rsp+40h] [rbp-210h] BYREF
  __int64 v91; // [rsp+48h] [rbp-208h]
  _BYTE v92[64]; // [rsp+50h] [rbp-200h] BYREF
  __int128 v93; // [rsp+90h] [rbp-1C0h] BYREF
  __int128 v94; // [rsp+A0h] [rbp-1B0h] BYREF
  __int128 *v95; // [rsp+B0h] [rbp-1A0h] BYREF
  __int64 v96; // [rsp+B8h] [rbp-198h]
  __int128 v97; // [rsp+C0h] [rbp-190h] BYREF
  __int128 v98; // [rsp+D0h] [rbp-180h]
  __int64 v99; // [rsp+E0h] [rbp-170h]
  __int64 v100; // [rsp+E8h] [rbp-168h]
  __int64 v101; // [rsp+F0h] [rbp-160h] BYREF
  __int64 *v102; // [rsp+F8h] [rbp-158h]
  __int64 *v103; // [rsp+100h] [rbp-150h]
  __int64 v104; // [rsp+108h] [rbp-148h]
  int v105; // [rsp+110h] [rbp-140h]
  _BYTE v106[312]; // [rsp+118h] [rbp-138h] BYREF

  v8 = v92;
  v9 = a1;
  v10 = *(unsigned int *)(a1 + 32168);
  v90 = v92;
  v11 = *(const void **)(a1 + 32160);
  v91 = 0x800000000LL;
  v12 = 8 * v10;
  v13 = v10;
  if ( v10 > 8 )
  {
    sub_16CD150((__int64)&v90, v92, v10, 8, a7, a8);
    v14 = &v90[8 * (unsigned int)v91];
  }
  else
  {
    v14 = v92;
    if ( !v12 )
      goto LABEL_3;
  }
  memcpy(v14, v11, v12);
  LODWORD(v12) = v91;
  v14 = v90;
LABEL_3:
  v15 = (__int64 *)v106;
  v101 = 0;
  LODWORD(v91) = v13 + v12;
  v16 = v13 + v12;
  v17 = (__int64 *)v106;
  v102 = (__int64 *)v106;
  v103 = (__int64 *)v106;
  v104 = 32;
  v105 = 0;
  if ( !(v13 + (_DWORD)v12) )
    goto LABEL_26;
LABEL_15:
  v25 = *(_QWORD *)&v14[8 * v16 - 8];
  LODWORD(v91) = v16 - 1;
  v88 = v25;
  if ( v17 != v15 )
    goto LABEL_5;
  v26 = &v17[HIDWORD(v104)];
  if ( v26 == v17 )
  {
LABEL_39:
    if ( HIDWORD(v104) < (unsigned int)v104 )
    {
      ++HIDWORD(v104);
      *v26 = v25;
      v15 = v102;
      ++v101;
      v17 = v103;
LABEL_6:
      v19 = v88;
      v20 = *(unsigned __int16 *)(v88 + 24);
      if ( (unsigned int)(v20 - 4) <= 1 || (unsigned __int16)(v20 - 7) <= 2u )
      {
        v21 = (unsigned int)v91;
        v22 = *(const void **)(v88 + 32);
        v23 = 8LL * *(_QWORD *)(v88 + 40);
        v24 = v23 >> 3;
        if ( v23 >> 3 > HIDWORD(v91) - (unsigned __int64)(unsigned int)v91 )
        {
          v84 = *(const void **)(v88 + 32);
          sub_16CD150((__int64)&v90, v8, v24 + (unsigned int)v91, 8, (int)v22, (int)v15);
          v21 = (unsigned int)v91;
          v22 = v84;
        }
        if ( v23 )
        {
          memcpy(&v90[8 * v21], v22, v23);
          LODWORD(v21) = v91;
        }
        v17 = v103;
        v15 = v102;
        LODWORD(v91) = v21 + v24;
        v16 = v21 + v24;
        goto LABEL_13;
      }
      if ( (unsigned __int16)(v20 - 1) <= 2u )
      {
        v29 = *(_QWORD *)(v88 + 32);
        v30 = (unsigned int)v91;
        if ( (unsigned int)v91 >= HIDWORD(v91) )
        {
          sub_16CD150((__int64)&v90, v8, 0, 8, (int)v17, (int)v15);
          v30 = (unsigned int)v91;
        }
        *(_QWORD *)&v90[8 * v30] = v29;
        v17 = v103;
        v15 = v102;
        v16 = v91 + 1;
        LODWORD(v91) = v91 + 1;
        goto LABEL_13;
      }
      if ( (_WORD)v20 == 6 )
      {
        *(_QWORD *)&v93 = *(_QWORD *)(v88 + 32);
        sub_1458920((__int64)&v90, &v93);
        *(_QWORD *)&v93 = *(_QWORD *)(v19 + 40);
        sub_1458920((__int64)&v90, &v93);
        v16 = v91;
        v17 = v103;
        v15 = v102;
        goto LABEL_13;
      }
      if ( (_WORD)v20 != 10 )
        goto LABEL_22;
      v31 = *(_QWORD *)(v88 - 8);
      v32 = *(_BYTE *)(v31 + 16);
      if ( v32 <= 0x17u )
      {
        if ( v32 == 9 )
          goto LABEL_22;
      }
      else if ( sub_1377F70(*(_QWORD *)(v9 + 40) + 56LL, *(_QWORD *)(v31 + 40)) )
      {
        v17 = v103;
        v15 = v102;
        goto LABEL_22;
      }
      v33 = *(_QWORD **)(v31 + 8);
      if ( !v33 )
        goto LABEL_76;
      v85 = v8;
      while ( 1 )
      {
        v34 = sub_1648700((__int64)v33);
        v35 = *((_BYTE *)v34 + 16);
        if ( v35 <= 0x17u )
          goto LABEL_48;
        v36 = v34[5];
        v37 = *(__int64 **)(*(_QWORD *)(v9 + 40) + 32LL);
        v38 = *v37;
        if ( *(_QWORD *)(*v37 + 56) != *(_QWORD *)(v36 + 56) )
          goto LABEL_48;
        if ( v35 == 77 )
        {
          v71 = sub_1648720((__int64)v33);
          if ( (*((_BYTE *)v34 + 23) & 0x40) != 0 )
            v72 = (_QWORD *)*(v34 - 1);
          else
            v72 = &v34[-3 * (*((_DWORD *)v34 + 5) & 0xFFFFFFF)];
          v36 = v72[3 * *((unsigned int *)v34 + 14) + 1 + v71];
          v38 = **(_QWORD **)(*(_QWORD *)(v9 + 40) + 32LL);
        }
        if ( !sub_15CC8F0(*(_QWORD *)(v9 + 16), v38, v36) )
          goto LABEL_48;
        v39 = (unsigned int)*(unsigned __int8 *)(sub_157EBA0(v36) + 16) - 34;
        if ( (unsigned int)v39 <= 0x36 )
        {
          v40 = 0x40018000000001LL;
          if ( _bittest64(&v40, v39) )
            goto LABEL_48;
        }
        if ( *(_BYTE *)(sub_157EBA0(v34[5]) + 16) == 34 )
          goto LABEL_48;
        if ( sub_1456C80(*(_QWORD *)(v9 + 8), *v34) )
        {
          v73 = sub_146F1B0(*(_QWORD *)(v9 + 8), (__int64)v34);
          if ( *(_WORD *)(v73 + 24) != 10 )
            goto LABEL_48;
          if ( v73 == v19 )
          {
            *(_QWORD *)&v93 = sub_145DC80(*(_QWORD *)(v9 + 8), (__int64)v34);
            sub_1458920((__int64)&v90, &v93);
            goto LABEL_48;
          }
        }
        if ( *((_BYTE *)v34 + 16) != 75
          || (v41 = sub_1648720((__int64)v33),
              v76 = *(_QWORD *)(v9 + 40),
              v81 = *(_QWORD *)(v9 + 8),
              v42 = sub_146F1B0(v81, v34[3 * (v41 == 0) - 6]),
              !sub_146D100(v81, v42, v76)) )
        {
          v8 = v85;
          v43 = sub_199DBD0(v9, &v88, 0, 0, 0xFFFFFFFFLL, a2, a3);
          a2 = 0;
          v44 = v34;
          v96 = 2;
          v86 = v43;
          v94 = 0;
          v93 = 0;
          v45 = *(_QWORD *)(v9 + 368) + 1984 * v43;
          *((_QWORD *)&v94 + 1) = (char *)&v97 + 8;
          v95 = (__int128 *)((char *)&v97 + 8);
          v97 = 0;
          v98 = 0;
          v82 = v46;
          v47 = *(_DWORD *)(v45 + 64);
          if ( v47 >= *(_DWORD *)(v45 + 68) )
          {
            v80 = v45;
            sub_19957D0((unsigned __int64 *)(v45 + 56), 0);
            v45 = v80;
            v44 = v34;
            v47 = *(_DWORD *)(v80 + 64);
          }
          v48 = 80LL * v47;
          v49 = *(_QWORD *)(v45 + 56) + v48 == 0;
          v50 = *(_QWORD *)(v45 + 56) + v48;
          v51 = v50;
          if ( !v49 )
          {
            v74 = v45;
            v77 = v44;
            *(_OWORD *)v50 = v93;
            sub_16CCEE0((_QWORD *)(v50 + 16), v50 + 56, 2, (__int64)&v94);
            v45 = v74;
            v44 = v77;
            *(_QWORD *)(v51 + 72) = *((_QWORD *)&v98 + 1);
            v47 = *(_DWORD *)(v74 + 64);
          }
          *(_DWORD *)(v45 + 64) = v47 + 1;
          if ( v95 != *((__int128 **)&v94 + 1) )
          {
            v75 = v45;
            v78 = v44;
            _libc_free((unsigned __int64)v95);
            v45 = v75;
            v44 = v78;
          }
          v79 = v45;
          v52 = *(_QWORD *)(v45 + 56) + 80LL * *(unsigned int *)(v45 + 64) - 80;
          *(_QWORD *)v52 = v44;
          *(_QWORD *)(v52 + 8) = *v33;
          *(_QWORD *)(v52 + 72) = v82;
          v53 = sub_19A2CE0((_QWORD *)v52, *(_QWORD *)(v9 + 40));
          v54 = v79;
          v55 = *(_QWORD *)(v79 + 736);
          *(_BYTE *)(v79 + 728) &= v53;
          if ( !v55
            || (v56 = sub_1456C90(*(_QWORD *)(v9 + 8), v55),
                v57 = sub_1456C90(*(_QWORD *)(v9 + 8), **(_QWORD **)(v52 + 8)),
                v54 = v79,
                v56 < v57) )
          {
            *(_QWORD *)(v54 + 736) = **(_QWORD **)(v52 + 8);
          }
          v83 = v54;
          v89 = v19;
          v96 = 0x400000000LL;
          v93 = 0u;
          LOBYTE(v94) = 0;
          *((_QWORD *)&v94 + 1) = 0;
          v95 = &v97;
          v99 = 0;
          v100 = 0;
          sub_1458920((__int64)&v95, &v89);
          LOBYTE(v94) = 1;
          sub_19A1660(v9, v83, v86, (__int64)&v93, v83, v58);
          v59 = v83;
          if ( v95 != &v97 )
          {
            _libc_free((unsigned __int64)v95);
            v59 = v83;
          }
          v60 = *(unsigned int *)(v9 + 376) - 1LL;
          v61 = *(_QWORD *)(v59 + 744) + 96LL * *(unsigned int *)(v59 + 752) - 96;
          v62 = *(_QWORD *)(v61 + 80);
          if ( v62 )
            sub_1998430(v9 + 32128, v62, *(unsigned int *)(v9 + 376) - 1LL);
          v63 = *(__int64 **)(v61 + 32);
          v64 = &v63[*(unsigned int *)(v61 + 40)];
          v65 = v63;
          if ( v63 != v64 )
          {
            v66 = v8;
            v87 = v9;
            v67 = v9 + 32128;
            v68 = v64;
            v69 = v66;
            do
            {
              v70 = *v65++;
              sub_1998430(v67, v70, v60);
            }
            while ( v68 != v65 );
            v9 = v87;
            v8 = v69;
          }
LABEL_76:
          v16 = v91;
          v17 = v103;
          v15 = v102;
LABEL_13:
          if ( !v16 )
            goto LABEL_23;
LABEL_14:
          v14 = v90;
          goto LABEL_15;
        }
LABEL_48:
        v33 = (_QWORD *)v33[1];
        if ( !v33 )
        {
          v8 = v85;
          goto LABEL_76;
        }
      }
    }
LABEL_5:
    sub_16CCBA0((__int64)&v101, v25);
    v17 = v103;
    v15 = v102;
    if ( !v18 )
      goto LABEL_22;
    goto LABEL_6;
  }
  v27 = v17;
  v28 = 0;
  while ( v25 != *v27 )
  {
    if ( *v27 == -2 )
      v28 = v27;
    if ( v26 == ++v27 )
    {
      if ( !v28 )
        goto LABEL_39;
      *v28 = v25;
      v17 = v103;
      --v105;
      v15 = v102;
      ++v101;
      goto LABEL_6;
    }
  }
LABEL_22:
  v16 = v91;
  if ( (_DWORD)v91 )
    goto LABEL_14;
LABEL_23:
  if ( v17 != v15 )
    _libc_free((unsigned __int64)v17);
  v14 = v90;
LABEL_26:
  if ( v14 != v8 )
    _libc_free((unsigned __int64)v14);
}
