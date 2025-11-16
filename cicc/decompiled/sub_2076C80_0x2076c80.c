// Function: sub_2076C80
// Address: 0x2076c80
//
void __fastcall sub_2076C80(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v7; // rax
  __int64 *v8; // r13
  __int64 v9; // r14
  __int64 v10; // rcx
  __int64 (*v11)(); // rdx
  __int64 v12; // rdi
  __int64 v13; // rax
  unsigned __int64 v14; // rdi
  unsigned int v15; // edx
  unsigned __int64 v16; // rdx
  __int64 *v17; // rax
  int v18; // r9d
  unsigned int v19; // r8d
  unsigned __int64 v20; // rdx
  __int64 v21; // r13
  unsigned __int64 v22; // rdx
  int v23; // r15d
  _BYTE *v24; // rax
  _BYTE *v25; // rdx
  int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rsi
  unsigned __int8 *v29; // rax
  unsigned int v30; // ecx
  unsigned int v31; // r8d
  unsigned __int16 v32; // r15
  __int64 v33; // rax
  __int64 (*v34)(); // rax
  unsigned int v35; // ebx
  unsigned int v36; // eax
  __int64 v37; // r12
  __int128 v38; // rax
  __int64 *v39; // rax
  __int64 v40; // r8
  __int64 v41; // rdx
  __int64 v42; // r9
  int v43; // edx
  _QWORD *v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rax
  int v47; // edx
  int v48; // edi
  __int64 v49; // rdx
  __int64 v50; // rax
  _BYTE *v51; // rax
  unsigned int v52; // edx
  __int64 *v53; // rax
  int v54; // edx
  __int64 v55; // r12
  __int64 *v56; // rbx
  int v57; // r14d
  char v58; // al
  __int16 v59; // ax
  __int128 v60; // [rsp-10h] [rbp-230h]
  __int128 v61; // [rsp-10h] [rbp-230h]
  const void **v62; // [rsp+0h] [rbp-220h]
  __int64 *v63; // [rsp+38h] [rbp-1E8h]
  __int64 v64; // [rsp+40h] [rbp-1E0h]
  unsigned int v65; // [rsp+4Ch] [rbp-1D4h]
  __int64 *v66; // [rsp+50h] [rbp-1D0h]
  unsigned __int64 v67; // [rsp+58h] [rbp-1C8h]
  __int64 *v68; // [rsp+60h] [rbp-1C0h]
  unsigned __int64 v69; // [rsp+68h] [rbp-1B8h]
  __int64 v70; // [rsp+70h] [rbp-1B0h]
  __int64 *v71; // [rsp+70h] [rbp-1B0h]
  unsigned int v72; // [rsp+78h] [rbp-1A8h]
  unsigned int v73; // [rsp+78h] [rbp-1A8h]
  unsigned int v74; // [rsp+78h] [rbp-1A8h]
  unsigned int v75; // [rsp+80h] [rbp-1A0h]
  unsigned int v76; // [rsp+80h] [rbp-1A0h]
  unsigned int v77; // [rsp+80h] [rbp-1A0h]
  __int64 v78; // [rsp+88h] [rbp-198h]
  unsigned int v79; // [rsp+88h] [rbp-198h]
  __int64 v80; // [rsp+88h] [rbp-198h]
  __int64 v81; // [rsp+D0h] [rbp-150h] BYREF
  int v82; // [rsp+D8h] [rbp-148h]
  __int64 v83[4]; // [rsp+E0h] [rbp-140h] BYREF
  __int128 v84; // [rsp+100h] [rbp-120h]
  __int64 v85; // [rsp+110h] [rbp-110h]
  _QWORD v86[2]; // [rsp+120h] [rbp-100h] BYREF
  _BYTE v87[32]; // [rsp+130h] [rbp-F0h] BYREF
  _BYTE *v88; // [rsp+150h] [rbp-D0h] BYREF
  __int64 v89; // [rsp+158h] [rbp-C8h]
  _BYTE v90[64]; // [rsp+160h] [rbp-C0h] BYREF
  _BYTE *v91; // [rsp+1A0h] [rbp-80h] BYREF
  __int64 v92; // [rsp+1A8h] [rbp-78h]
  _BYTE v93[112]; // [rsp+1B0h] [rbp-70h] BYREF

  if ( sub_15F32D0(a2) )
  {
    sub_2076A60(a1, a2, a3, a4, a5);
    return;
  }
  v7 = *(_QWORD *)(a1 + 552);
  v8 = *(__int64 **)(a2 - 48);
  v9 = *(_QWORD *)(a2 - 24);
  v10 = *(_QWORD *)(v7 + 16);
  v70 = v10;
  v11 = *(__int64 (**)())(*(_QWORD *)v10 + 1160LL);
  if ( v11 != sub_1D45FE0 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64))v11)(v10) )
    {
      v58 = *(_BYTE *)(v9 + 16);
      if ( v58 == 17 )
      {
        if ( (unsigned __int8)sub_15E02D0(v9) )
        {
LABEL_54:
          sub_2074630(a1, a2, a3, a4, a5);
          return;
        }
        v58 = *(_BYTE *)(v9 + 16);
      }
      if ( v58 == 53 && (*(_BYTE *)(v9 + 18) & 0x40) != 0 )
        goto LABEL_54;
    }
    v7 = *(_QWORD *)(a1 + 552);
  }
  v12 = *(_QWORD *)(v7 + 32);
  v88 = v90;
  v86[0] = v87;
  v89 = 0x400000000LL;
  v86[1] = 0x400000000LL;
  v78 = *v8;
  v13 = sub_1E0A0C0(v12);
  sub_20C7CE0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL), v13, v78, &v88, v86, 0);
  if ( !(_DWORD)v89 )
    goto LABEL_4;
  v75 = v89;
  v63 = sub_20685E0(a1, v8, a3, a4, a5);
  v79 = v15;
  v66 = sub_20685E0(a1, (__int64 *)v9, a3, a4, a5);
  v67 = v16;
  v17 = sub_2051C20((__int64 *)a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
  v19 = v75;
  v68 = v17;
  v69 = v20;
  if ( v75 <= 0x3F )
  {
    v22 = v75;
    v92 = 0x400000000LL;
    v23 = v75;
    v91 = v93;
    v21 = 16LL * v75;
    if ( v75 <= 4 )
    {
      v24 = v93;
      goto LABEL_14;
    }
  }
  else
  {
    v92 = 0x400000000LL;
    v21 = 1024;
    v22 = 64;
    v23 = 64;
    v91 = v93;
  }
  sub_16CD150((__int64)&v91, v93, v22, 16, v75, v18);
  v24 = v91;
  v19 = v75;
LABEL_14:
  LODWORD(v92) = v23;
  v25 = &v24[v21];
  do
  {
    if ( v24 )
    {
      *(_QWORD *)v24 = 0;
      *((_DWORD *)v24 + 2) = 0;
    }
    v24 += 16;
  }
  while ( v25 != v24 );
  v26 = *(_DWORD *)(a1 + 536);
  v27 = *(_QWORD *)a1;
  v81 = 0;
  v82 = v26;
  if ( v27 )
  {
    if ( &v81 != (__int64 *)(v27 + 48) )
    {
      v28 = *(_QWORD *)(v27 + 48);
      v81 = v28;
      if ( v28 )
      {
        v76 = v19;
        sub_1623A60((__int64)&v81, v28, 2);
        v19 = v76;
      }
    }
  }
  v72 = v19;
  v29 = (unsigned __int8 *)(v66[5] + 16LL * (unsigned int)v67);
  v62 = (const void **)*((_QWORD *)v29 + 1);
  v30 = *v29;
  memset(v83, 0, 24);
  v77 = v30;
  v65 = 1 << (*(unsigned __int16 *)(a2 + 18) >> 1) >> 1;
  sub_14A8180(a2, v83, 0);
  v31 = v72;
  v32 = 4 * (*(_WORD *)(a2 + 18) & 1);
  if ( *(_QWORD *)(a2 + 48) || *(__int16 *)(a2 + 18) < 0 )
  {
    v33 = sub_1625790(a2, 9);
    v31 = v72;
    if ( v33 )
      v32 |= 8u;
  }
  v34 = *(__int64 (**)())(*(_QWORD *)v70 + 1296LL);
  if ( v34 != sub_2043C70 )
  {
    v74 = v31;
    v59 = ((__int64 (__fastcall *)(__int64, __int64))v34)(v70, a2);
    v31 = v74;
    v32 |= v59;
  }
  v35 = 0;
  v36 = v79;
  v80 = a1;
  v37 = 0;
  v73 = v36;
  v64 = 8LL * (v31 - 1);
  while ( 1 )
  {
    v71 = *(__int64 **)(v80 + 552);
    *(_QWORD *)&v38 = sub_1D38BB0(
                        (__int64)v71,
                        *(_QWORD *)(v86[0] + v37),
                        (__int64)&v81,
                        v77,
                        v62,
                        0,
                        a3,
                        *(double *)a4.m128i_i64,
                        a5,
                        0);
    v39 = sub_1D332F0(
            v71,
            52,
            (__int64)&v81,
            v77,
            v62,
            3u,
            *(double *)a3.m128i_i64,
            *(double *)a4.m128i_i64,
            a5,
            (__int64)v66,
            v67,
            v38);
    *(_QWORD *)&v84 = v9;
    v40 = (__int64)v39;
    v42 = v41;
    v43 = 0;
    LOBYTE(v85) = 0;
    v44 = *(_QWORD **)(v80 + 552);
    *((_QWORD *)&v84 + 1) = *(_QWORD *)(v86[0] + v37);
    if ( v9 )
    {
      v45 = *(_QWORD *)v9;
      if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) == 16 )
        v45 = **(_QWORD **)(v45 + 16);
      v43 = *(_DWORD *)(v45 + 8) >> 8;
    }
    HIDWORD(v85) = v43;
    v46 = sub_1D2BF40(
            v44,
            (__int64)v68,
            v69,
            (__int64)&v81,
            (__int64)v63,
            v73,
            v40,
            v42,
            v84,
            v85,
            v65,
            v32,
            (__int64)v83);
    v48 = v47;
    v49 = v46;
    v50 = v35++;
    v51 = &v91[16 * v50];
    *(_QWORD *)v51 = v49;
    *((_DWORD *)v51 + 2) = v48;
    if ( v37 == v64 )
      break;
    if ( v35 == 64 )
    {
      v35 = 0;
      *((_QWORD *)&v60 + 1) = 64;
      *(_QWORD *)&v60 = v91;
      v68 = sub_1D359D0(
              *(__int64 **)(v80 + 552),
              2,
              (__int64)&v81,
              1,
              0,
              0,
              *(double *)a3.m128i_i64,
              *(double *)a4.m128i_i64,
              a5,
              v60);
      v69 = v52 | v69 & 0xFFFFFFFF00000000LL;
    }
    ++v73;
    v37 += 8;
  }
  *((_QWORD *)&v61 + 1) = v35;
  *(_QWORD *)&v61 = v91;
  v53 = sub_1D359D0(
          *(__int64 **)(v80 + 552),
          2,
          (__int64)&v81,
          1,
          0,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          a5,
          v61);
  v55 = *(_QWORD *)(v80 + 552);
  v56 = v53;
  v57 = v54;
  if ( v53 )
  {
    nullsub_686();
    *(_QWORD *)(v55 + 176) = v56;
    *(_DWORD *)(v55 + 184) = v57;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v55 + 176) = 0;
    *(_DWORD *)(v55 + 184) = v54;
  }
  if ( v81 )
    sub_161E7C0((__int64)&v81, v81);
  if ( v91 == v93 )
  {
LABEL_4:
    v14 = v86[0];
    if ( (_BYTE *)v86[0] == v87 )
      goto LABEL_6;
    goto LABEL_5;
  }
  _libc_free((unsigned __int64)v91);
  v14 = v86[0];
  if ( (_BYTE *)v86[0] != v87 )
LABEL_5:
    _libc_free(v14);
LABEL_6:
  if ( v88 != v90 )
    _libc_free((unsigned __int64)v88);
}
