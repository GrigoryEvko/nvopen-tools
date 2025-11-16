// Function: sub_33A1620
// Address: 0x33a1620
//
void __fastcall sub_33A1620(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // r13
  __int64 (*v10)(); // rax
  __int64 *v11; // rdi
  __int64 v12; // r15
  int v13; // eax
  unsigned int v14; // ebx
  unsigned __int64 v15; // rdi
  unsigned int v16; // edx
  unsigned int v17; // r15d
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r9
  int v23; // edx
  __int64 v24; // r8
  unsigned __int64 v25; // rdx
  int v26; // r14d
  _BYTE *v27; // rdx
  _BYTE *v28; // rax
  _BYTE *i; // r8
  int v30; // edx
  __int64 v31; // rax
  __int64 v32; // rsi
  unsigned __int64 v33; // rax
  __int64 v34; // rdi
  unsigned int v35; // r13d
  __int64 v36; // rax
  __int64 v37; // rbx
  __int64 v38; // rax
  int v39; // r9d
  int v40; // edx
  int v41; // edi
  __int64 v42; // rdx
  __int64 v43; // rax
  _BYTE *v44; // rax
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rdx
  int v48; // r8d
  __int64 v49; // r10
  __int64 v50; // r11
  unsigned int v51; // r9d
  unsigned int *v52; // rax
  __int16 v53; // cx
  int v54; // eax
  unsigned int v55; // edx
  int v56; // edx
  int v57; // edx
  __int64 v58; // rdx
  int v59; // edx
  __int64 v60; // r13
  int v61; // edx
  int v62; // ebx
  _QWORD *v63; // rax
  __int64 v64; // r12
  char v65; // al
  __int128 v66; // [rsp-10h] [rbp-2F0h]
  __int128 v67; // [rsp-10h] [rbp-2F0h]
  int v68; // [rsp+40h] [rbp-2A0h]
  int v69; // [rsp+48h] [rbp-298h]
  char v70; // [rsp+5Bh] [rbp-285h]
  __int16 v71; // [rsp+5Ch] [rbp-284h]
  int v72; // [rsp+60h] [rbp-280h]
  __int64 v73; // [rsp+60h] [rbp-280h]
  __int64 v74; // [rsp+68h] [rbp-278h]
  __int64 v75; // [rsp+70h] [rbp-270h]
  __int64 v76; // [rsp+70h] [rbp-270h]
  int v77; // [rsp+78h] [rbp-268h]
  __int64 v79; // [rsp+88h] [rbp-258h]
  int v80; // [rsp+90h] [rbp-250h]
  __int64 v81; // [rsp+98h] [rbp-248h]
  __int64 v82; // [rsp+100h] [rbp-1E0h] BYREF
  int v83; // [rsp+108h] [rbp-1D8h]
  __int64 v84; // [rsp+110h] [rbp-1D0h]
  unsigned __int64 v85; // [rsp+118h] [rbp-1C8h]
  __m128i v86; // [rsp+120h] [rbp-1C0h]
  __int128 v87; // [rsp+130h] [rbp-1B0h] BYREF
  __int64 v88; // [rsp+140h] [rbp-1A0h]
  __int64 v89[4]; // [rsp+150h] [rbp-190h] BYREF
  _BYTE *v90; // [rsp+170h] [rbp-170h] BYREF
  __int64 v91; // [rsp+178h] [rbp-168h]
  _BYTE v92[64]; // [rsp+180h] [rbp-160h] BYREF
  unsigned __int64 v93[2]; // [rsp+1C0h] [rbp-120h] BYREF
  _BYTE v94[64]; // [rsp+1D0h] [rbp-110h] BYREF
  _QWORD v95[2]; // [rsp+210h] [rbp-D0h] BYREF
  _BYTE v96[64]; // [rsp+220h] [rbp-C0h] BYREF
  _BYTE *v97; // [rsp+260h] [rbp-80h] BYREF
  __int64 v98; // [rsp+268h] [rbp-78h]
  _BYTE v99[112]; // [rsp+270h] [rbp-70h] BYREF

  if ( sub_B46500(a2) )
  {
    sub_33A10A0(a1, (__int64)a2, v3, v4, v5, v6);
    return;
  }
  v7 = *(_QWORD *)(a1 + 864);
  v8 = *((_QWORD *)a2 - 8);
  v9 = *(_QWORD *)(v7 + 16);
  v79 = *((_QWORD *)a2 - 4);
  v10 = *(__int64 (**)())(*(_QWORD *)v9 + 2216LL);
  if ( v10 != sub_302E1B0 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64))v10)(v9) )
    {
      v65 = *(_BYTE *)v79;
      if ( *(_BYTE *)v79 == 22 )
      {
        if ( (unsigned __int8)sub_B2D650(v79) )
        {
LABEL_60:
          sub_339CB00(a1, (__int64)a2);
          return;
        }
        v65 = *(_BYTE *)v79;
      }
      if ( v65 == 60 && *(char *)(v79 + 2) < 0 )
        goto LABEL_60;
    }
    v7 = *(_QWORD *)(a1 + 864);
  }
  v11 = *(__int64 **)(v7 + 40);
  v84 = 0;
  v90 = v92;
  v93[0] = (unsigned __int64)v94;
  v91 = 0x400000000LL;
  v93[1] = 0x400000000LL;
  v95[0] = v96;
  v95[1] = 0x400000000LL;
  v12 = *(_QWORD *)(v8 + 8);
  LOBYTE(v85) = 0;
  v13 = sub_2E79000(v11);
  sub_34B8C80(
    *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL),
    v13,
    v12,
    (unsigned int)&v90,
    (unsigned int)v93,
    (unsigned int)v95,
    __PAIR128__(v85, 0));
  v14 = v91;
  if ( !(_DWORD)v91 )
    goto LABEL_4;
  v81 = sub_338B750(a1, v8);
  v17 = v16;
  v72 = sub_338B750(a1, v79);
  v68 = v18;
  if ( (a2[2] & 1) != 0 )
  {
    v77 = sub_33738B0(a1, v79, v18, v19, v20, v21);
    v80 = v56;
    if ( v14 > 0x3F )
      goto LABEL_13;
  }
  else
  {
    v77 = sub_33738A0(a1);
    v80 = v23;
    if ( v14 > 0x3F )
    {
LABEL_13:
      v24 = 1024;
      v25 = 64;
      v26 = 64;
      v97 = v99;
      v98 = 0x400000000LL;
LABEL_14:
      v75 = v24;
      sub_C8D5F0((__int64)&v97, v99, v25, 0x10u, v24, v22);
      v27 = v97;
      v24 = v75;
      v28 = &v97[16 * (unsigned int)v98];
      goto LABEL_15;
    }
  }
  v25 = v14;
  v28 = v99;
  v26 = v14;
  v97 = v99;
  v24 = 16LL * v14;
  v98 = 0x400000000LL;
  if ( v14 > 4 )
    goto LABEL_14;
  v27 = v99;
LABEL_15:
  for ( i = &v27[v24]; i != v28; v28 += 16 )
  {
    if ( v28 )
    {
      *(_QWORD *)v28 = 0;
      *((_DWORD *)v28 + 2) = 0;
    }
  }
  v30 = *(_DWORD *)(a1 + 848);
  v31 = *(_QWORD *)a1;
  LODWORD(v98) = v26;
  v82 = 0;
  v83 = v30;
  if ( v31 )
  {
    if ( &v82 != (__int64 *)(v31 + 48) )
    {
      v32 = *(_QWORD *)(v31 + 48);
      v82 = v32;
      if ( v32 )
        sub_B96E90((__int64)&v82, v32, 1);
    }
  }
  _BitScanReverse64(&v33, 1LL << (*((_WORD *)a2 + 1) >> 1));
  v70 = 63 - (v33 ^ 0x3F);
  sub_B91FC0(v89, (__int64)a2);
  sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  v34 = v9;
  v35 = 0;
  v71 = sub_2FEC5A0(v34, (__int64)a2);
  v36 = v14 - 1;
  v37 = 0;
  v76 = 16 * v36;
  v69 = v72;
  while ( 1 )
  {
    if ( *(_BYTE *)(v37 + v95[0] + 8) && *(_QWORD *)(v37 + v95[0]) )
    {
      v87 = 0u;
      LODWORD(v88) = 0;
      BYTE4(v88) = 0;
    }
    else
    {
      *((_QWORD *)&v87 + 1) = *(_QWORD *)(v37 + v95[0]);
      v57 = 0;
      BYTE4(v88) = 0;
      *(_QWORD *)&v87 = v79 & 0xFFFFFFFFFFFFFFFBLL;
      if ( v79 )
      {
        v58 = *(_QWORD *)(v79 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v58 + 8) - 17 <= 1 )
          v58 = **(_QWORD **)(v58 + 16);
        v57 = *(_DWORD *)(v58 + 8) >> 8;
      }
      LODWORD(v88) = v57;
    }
    v45 = *(_QWORD *)(a1 + 864);
    v86 = _mm_loadu_si128((const __m128i *)(v37 + v95[0]));
    v46 = sub_3409320(v45, v69, v68, v86.m128i_i32[0], v86.m128i_i32[2], (unsigned int)&v82, 1);
    v48 = v81;
    v49 = v46;
    v50 = v47;
    v51 = v17;
    v52 = (unsigned int *)(v37 + v93[0]);
    v53 = *(_WORD *)(v37 + v93[0]);
    if ( v53 != *(_WORD *)&v90[v37] || !v53 && *(_QWORD *)&v90[v37 + 8] != *((_QWORD *)v52 + 1) )
    {
      v73 = v49;
      v74 = v47;
      v54 = sub_33FB4C0(*(_QWORD *)(a1 + 864), v81, v17, &v82, *v52, *((_QWORD *)v52 + 1));
      v49 = v73;
      v50 = v74;
      v48 = v54;
      v51 = v55;
    }
    v38 = sub_33F4560(
            *(_QWORD *)(a1 + 864),
            v77,
            v80,
            (unsigned int)&v82,
            v48,
            v51,
            v49,
            v50,
            v87,
            v88,
            v70,
            v71,
            (__int64)v89);
    v41 = v40;
    v42 = v38;
    v43 = v35++;
    v44 = &v97[16 * v43];
    *(_QWORD *)v44 = v42;
    *((_DWORD *)v44 + 2) = v41;
    if ( v37 == v76 )
      break;
    if ( v35 == 64 )
    {
      v35 = 0;
      *((_QWORD *)&v66 + 1) = 64;
      *(_QWORD *)&v66 = v97;
      v77 = sub_33FC220(*(_QWORD *)(a1 + 864), 2, (unsigned int)&v82, 1, 0, v39, v66);
      v80 = v59;
    }
    v37 += 16;
    ++v17;
  }
  *((_QWORD *)&v67 + 1) = v35;
  *(_QWORD *)&v67 = v97;
  v60 = sub_33FC220(*(_QWORD *)(a1 + 864), 2, (unsigned int)&v82, 1, 0, v39, v67);
  v62 = v61;
  *(_QWORD *)&v87 = a2;
  v63 = sub_337DC20(a1 + 8, (__int64 *)&v87);
  *v63 = v60;
  *((_DWORD *)v63 + 2) = v62;
  v64 = *(_QWORD *)(a1 + 864);
  if ( v60 )
  {
    nullsub_1875(v60, v64, 0);
    *(_QWORD *)(v64 + 384) = v60;
    *(_DWORD *)(v64 + 392) = v62;
    sub_33E2B60(v64, 0);
  }
  else
  {
    *(_QWORD *)(v64 + 384) = 0;
    *(_DWORD *)(v64 + 392) = v62;
  }
  if ( v82 )
    sub_B91220((__int64)&v82, v82);
  if ( v97 != v99 )
  {
    _libc_free((unsigned __int64)v97);
    v15 = v95[0];
    if ( (_BYTE *)v95[0] == v96 )
      goto LABEL_6;
    goto LABEL_5;
  }
LABEL_4:
  v15 = v95[0];
  if ( (_BYTE *)v95[0] != v96 )
LABEL_5:
    _libc_free(v15);
LABEL_6:
  if ( (_BYTE *)v93[0] != v94 )
    _libc_free(v93[0]);
  if ( v90 != v92 )
    _libc_free((unsigned __int64)v90);
}
