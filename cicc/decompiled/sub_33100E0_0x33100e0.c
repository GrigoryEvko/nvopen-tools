// Function: sub_33100E0
// Address: 0x33100e0
//
__int64 __fastcall sub_33100E0(_QWORD *a1, __int64 a2)
{
  __m128i v4; // xmm0
  __int16 *v5; // rax
  __int64 v6; // rsi
  unsigned __int16 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r9
  bool v11; // zf
  __int64 v12; // rax
  int v13; // r9d
  __int64 v14; // r15
  int v16; // esi
  int v17; // edx
  int v18; // r9d
  int v19; // eax
  unsigned __int8 v20; // si
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r15
  __int64 v29; // r14
  __int128 v30; // rax
  _QWORD *v31; // rdx
  __int64 v32; // rax
  unsigned __int16 *v33; // rdx
  _QWORD *v34; // rax
  __int64 v35; // r14
  __int64 v36; // rdx
  __int64 v37; // r15
  int v38; // r9d
  __int128 v39; // rax
  int v40; // r9d
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r14
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rax
  char v47; // al
  __int64 *v48; // r10
  char v49; // al
  __int64 v50; // rdi
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rcx
  __int64 v54; // rax
  __int16 v55; // r12
  __int64 v56; // r14
  __int16 v57; // ax
  __int64 v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // rdx
  unsigned int v61; // eax
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r15
  __int64 v68; // r14
  __int64 v69; // rdx
  __int128 v70; // [rsp-20h] [rbp-130h]
  __int128 v71; // [rsp-20h] [rbp-130h]
  __int64 v72; // [rsp+0h] [rbp-110h]
  __int128 v73; // [rsp+0h] [rbp-110h]
  __int64 v74; // [rsp+10h] [rbp-100h]
  __int64 v75; // [rsp+20h] [rbp-F0h]
  char v76; // [rsp+30h] [rbp-E0h]
  __int64 v77; // [rsp+30h] [rbp-E0h]
  unsigned int v78; // [rsp+38h] [rbp-D8h]
  __int64 v79; // [rsp+38h] [rbp-D8h]
  int v80; // [rsp+38h] [rbp-D8h]
  unsigned int v81; // [rsp+40h] [rbp-D0h]
  __int64 v82; // [rsp+40h] [rbp-D0h]
  __int64 v83; // [rsp+48h] [rbp-C8h]
  __m128i v84; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v85; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v86; // [rsp+68h] [rbp-A8h]
  __int64 v87; // [rsp+70h] [rbp-A0h] BYREF
  int v88; // [rsp+78h] [rbp-98h]
  unsigned int v89; // [rsp+80h] [rbp-90h] BYREF
  __int64 v90; // [rsp+88h] [rbp-88h]
  __int64 v91; // [rsp+90h] [rbp-80h]
  __int64 v92; // [rsp+98h] [rbp-78h]
  __int64 v93; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v94; // [rsp+A8h] [rbp-68h]
  _QWORD *v95; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v96; // [rsp+B8h] [rbp-58h]
  _QWORD v97[10]; // [rsp+C0h] [rbp-50h] BYREF

  v4 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v5 = *(__int16 **)(a2 + 48);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = *v5;
  v8 = *((_QWORD *)v5 + 1);
  v84 = v4;
  v87 = v6;
  LOWORD(v85) = v7;
  v86 = v8;
  if ( v6 )
    sub_B96E90((__int64)&v87, v6, 1);
  v9 = v84.m128i_i64[0];
  v10 = *a1;
  v11 = *(_DWORD *)(v84.m128i_i64[0] + 24) == 51;
  v88 = *(_DWORD *)(a2 + 72);
  if ( v11 )
  {
    v95 = 0;
    LODWORD(v96) = 0;
    v14 = sub_33F17F0(v10, 51, &v95, v85, v86);
    if ( v95 )
      sub_B91220((__int64)&v95, (__int64)v95);
    goto LABEL_6;
  }
  v12 = sub_32788C0(a2, (int)&v87, a1[1], v10, *((_BYTE *)a1 + 34), v10);
  if ( v12 )
  {
LABEL_5:
    v14 = v12;
    goto LABEL_6;
  }
  v16 = *(_DWORD *)(v84.m128i_i64[0] + 24);
  v17 = v16;
  if ( v16 == 215 )
  {
    v18 = 0;
LABEL_16:
    v14 = sub_33FA050(
            *a1,
            v16,
            (unsigned int)&v87,
            v85,
            v86,
            v18,
            **(_QWORD **)(v84.m128i_i64[0] + 40),
            *(_QWORD *)(*(_QWORD *)(v84.m128i_i64[0] + 40) + 8LL));
    goto LABEL_6;
  }
  if ( (unsigned int)(v16 - 213) <= 1 )
  {
    v18 = 0;
    if ( v16 == 214 )
    {
      v18 = *(_DWORD *)(v84.m128i_i64[0] + 28) & 0x10;
      if ( v18 )
        v18 = 16;
    }
    goto LABEL_16;
  }
  if ( (unsigned int)(v16 - 223) <= 2 )
  {
    v14 = sub_33FAF80(*a1, v16, (unsigned int)&v87, v85, v86, v13, *(_OWORD *)*(_QWORD *)(v84.m128i_i64[0] + 40));
    goto LABEL_6;
  }
  if ( v16 == 216 )
  {
    v41 = sub_32B3F40(a1, v84.m128i_i64[0]);
    if ( v41 )
    {
      if ( v41 != v84.m128i_i64[0] )
      {
        v43 = **(_QWORD **)(v84.m128i_i64[0] + 40);
        v96 = v42;
        v95 = (_QWORD *)v41;
        sub_32EB790((__int64)a1, v84.m128i_i64[0], (__int64 *)&v95, 1, 1);
        sub_32B3E80((__int64)a1, v43, 1, 0, v44, v45);
      }
      goto LABEL_33;
    }
    v17 = *(_DWORD *)(v84.m128i_i64[0] + 24);
    if ( v17 == 216 )
    {
      v12 = sub_33FAFB0(
              *a1,
              **(_QWORD **)(v84.m128i_i64[0] + 40),
              *(_QWORD *)(*(_QWORD *)(v84.m128i_i64[0] + 40) + 8LL),
              &v87,
              (unsigned int)v85,
              v86);
      goto LABEL_5;
    }
  }
  if ( v17 == 186 )
  {
    v31 = *(_QWORD **)(v84.m128i_i64[0] + 40);
    v32 = *v31;
    if ( *(_DWORD *)(*v31 + 24LL) == 216 && *(_DWORD *)(v31[5] + 24LL) == 11 )
    {
      v33 = (unsigned __int16 *)(*(_QWORD *)(v84.m128i_i64[0] + 48) + 16LL * v84.m128i_u32[2]);
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(*(_QWORD *)a1[1] + 1408LL))(
              a1[1],
              **(_QWORD **)(v32 + 40),
              *(_QWORD *)(*(_QWORD *)(v32 + 40) + 8LL),
              *v33,
              *((_QWORD *)v33 + 1)) )
      {
        v34 = *(_QWORD **)(**(_QWORD **)(v84.m128i_i64[0] + 40) + 40LL);
        v35 = sub_33FAFB0(*a1, *v34, v34[1], &v87, (unsigned int)v85, v86);
        v37 = v36;
        *(_QWORD *)&v39 = sub_33FAF80(
                            *a1,
                            215,
                            (unsigned int)&v87,
                            v85,
                            v86,
                            v38,
                            *(_OWORD *)(*(_QWORD *)(v84.m128i_i64[0] + 40) + 40LL));
        *((_QWORD *)&v71 + 1) = v37;
        *(_QWORD *)&v71 = v35;
        v14 = sub_3406EB0(*a1, 186, (unsigned int)&v87, v85, v86, v40, v71, v39);
        goto LABEL_6;
      }
    }
  }
  if ( !v7 )
  {
    if ( !sub_30070B0((__int64)&v85) )
      goto LABEL_24;
LABEL_35:
    v12 = sub_330B820(
            *a1,
            a1,
            a1[1],
            (unsigned int)v85,
            v86,
            *((_BYTE *)a1 + 33),
            a2,
            v84.m128i_i64[0],
            v84.m128i_i64[1],
            3,
            0xD6u,
            0);
    if ( v12 )
      goto LABEL_5;
    goto LABEL_36;
  }
  if ( (unsigned __int16)(v7 - 17) <= 0xD3u )
    goto LABEL_35;
LABEL_24:
  v19 = *(_DWORD *)(v84.m128i_i64[0] + 24);
  if ( v19 != 298 )
    goto LABEL_37;
  v20 = (*(_BYTE *)(v84.m128i_i64[0] + 33) >> 2) & 3;
  if ( v20 )
  {
LABEL_26:
    if ( (*(_WORD *)(v9 + 32) & 0x380) == 0 )
    {
      if ( (unsigned __int8)sub_3286E00(&v84) )
      {
        v21 = *(unsigned __int16 *)(v9 + 96);
        if ( !*((_BYTE *)a1 + 33)
          || (_WORD)v21
          && v7
          && (((int)*(unsigned __int16 *)(a1[1] + 2 * (v21 + 274LL * v7 + 71704) + 6) >> (4 * v20)) & 0xF) == 0 )
        {
          v22 = sub_33F1B30(
                  *a1,
                  v20,
                  (unsigned int)&v87,
                  v85,
                  v86,
                  *(_QWORD *)(v9 + 112),
                  **(_QWORD **)(v9 + 40),
                  *(_QWORD *)(*(_QWORD *)(v9 + 40) + 8LL),
                  *(_QWORD *)(*(_QWORD *)(v9 + 40) + 40LL),
                  *(_QWORD *)(*(_QWORD *)(v9 + 40) + 48LL),
                  *(unsigned __int16 *)(v9 + 96),
                  *(_QWORD *)(v9 + 104));
          v96 = v23;
          v24 = v22;
          v95 = (_QWORD *)v22;
          sub_32EB790((__int64)a1, a2, (__int64 *)&v95, 1, 1);
          sub_34161C0(*a1, v9, 1, v24, 1);
          sub_32CF870((__int64)a1, v9);
LABEL_33:
          v14 = a2;
          goto LABEL_6;
        }
      }
    }
    goto LABEL_44;
  }
  if ( (*(_WORD *)(v84.m128i_i64[0] + 32) & 0x380) != 0 )
    goto LABEL_44;
  v81 = v84.m128i_u32[2];
  v79 = 16LL * v84.m128i_u32[2];
  v46 = *(unsigned __int16 *)(*(_QWORD *)(v84.m128i_i64[0] + 48) + v79);
  if ( !v7 )
    goto LABEL_44;
  if ( !(_WORD)v46 )
    goto LABEL_44;
  v74 = a1[1];
  if ( (unsigned __int8)*(_WORD *)(v74 + 2 * (v46 + 274LL * v7 + 71704) + 6) >> 4 )
    goto LABEL_44;
  v95 = v97;
  v96 = 0x400000000LL;
  v47 = sub_3286E00(&v84);
  LODWORD(v48) = (unsigned int)&v87;
  if ( v47 || (v49 = sub_32611B0(v85, v86, a2, v9, v81, 215, (__int64)&v95, v74), v48 = &v87, v49) )
  {
    v65 = sub_33F1B30(
            *a1,
            1,
            (_DWORD)v48,
            v85,
            v86,
            *(_QWORD *)(v9 + 112),
            **(_QWORD **)(v9 + 40),
            *(_QWORD *)(*(_QWORD *)(v9 + 40) + 8LL),
            *(_QWORD *)(*(_QWORD *)(v9 + 40) + 40LL),
            *(_QWORD *)(*(_QWORD *)(v9 + 40) + 48LL),
            *(unsigned __int16 *)(*(_QWORD *)(v9 + 48) + v79),
            *(_QWORD *)(*(_QWORD *)(v9 + 48) + v79 + 8));
    v67 = v66;
    v68 = v65;
    sub_3304760(a1, (__int64)&v95, v9, v81, v65, v66, 215);
    v76 = sub_3286E00(&v84);
    v93 = v68;
    v94 = v67;
    sub_32EB790((__int64)a1, a2, &v93, 1, 1);
    if ( v76 )
    {
      sub_34161C0(*a1, v9, 1, v68, 1);
      sub_32CF870((__int64)a1, v9);
    }
    else
    {
      v75 = *a1;
      v77 = *(_QWORD *)(*(_QWORD *)(v9 + 48) + v79 + 8);
      v80 = *(unsigned __int16 *)(*(_QWORD *)(v9 + 48) + v79);
      sub_3285E70((__int64)&v93, v84.m128i_i64[0]);
      *((_QWORD *)&v73 + 1) = v67;
      *(_QWORD *)&v73 = v68;
      v82 = sub_33FAF80(v75, 216, (unsigned int)&v93, v80, v77, (unsigned int)&v93, v73);
      v83 = v69;
      sub_9C6650(&v93);
      sub_32EFDE0((__int64)a1, v9, v82, v83, v68, 1, 1);
    }
    v14 = a2;
    if ( v95 != v97 )
      _libc_free((unsigned __int64)v95);
    goto LABEL_6;
  }
  if ( v95 != v97 )
    _libc_free((unsigned __int64)v95);
LABEL_36:
  v19 = *(_DWORD *)(v9 + 24);
  if ( v19 == 298 )
  {
    v20 = (*(_BYTE *)(v9 + 33) >> 2) & 3;
    if ( v20 )
      goto LABEL_26;
    goto LABEL_44;
  }
LABEL_37:
  if ( v19 != 208 )
    goto LABEL_44;
  v25 = *a1;
  LODWORD(v96) = *(_DWORD *)(v9 + 28);
  v95 = (_QWORD *)v25;
  v97[0] = *(_QWORD *)(v25 + 1024);
  *(_QWORD *)(v25 + 1024) = &v95;
  if ( v7 )
  {
    if ( (unsigned __int16)(v7 - 17) <= 0xD3u )
      goto LABEL_40;
  }
  else if ( sub_30070B0((__int64)&v85) )
  {
LABEL_40:
    if ( !*((_BYTE *)a1 + 33) )
    {
      v50 = *a1;
      v51 = a1[1];
      v52 = *(_QWORD *)(**(_QWORD **)(v9 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v9 + 40) + 8LL);
      v53 = *(_QWORD *)(v52 + 8);
      LOWORD(v89) = *(_WORD *)v52;
      v54 = *(_QWORD *)(v9 + 48) + 16LL * v84.m128i_u32[2];
      v90 = v53;
      v55 = *(_WORD *)v54;
      v56 = *(_QWORD *)(v54 + 8);
      v57 = sub_325F2E0(v50, v51, v89, v53);
      if ( v57 == v55 && (v57 || v58 == v56) )
      {
        v14 = 0;
      }
      else
      {
        v93 = sub_2D5B750((unsigned __int16 *)&v89);
        v94 = v59;
        v91 = sub_2D5B750((unsigned __int16 *)&v85);
        v92 = v60;
        if ( v91 == v93 && (_BYTE)v92 == (_BYTE)v94 )
        {
          v14 = sub_32889F0(
                  *a1,
                  (int)&v87,
                  (unsigned int)v85,
                  v86,
                  **(_QWORD **)(v9 + 40),
                  *(_QWORD *)(*(_QWORD *)(v9 + 40) + 8LL),
                  *(_OWORD *)(*(_QWORD *)(v9 + 40) + 40LL),
                  *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v9 + 40) + 80LL) + 96LL),
                  0);
        }
        else
        {
          v61 = sub_327FDF0((unsigned __int16 *)&v89, v51);
          v63 = sub_32889F0(
                  *a1,
                  (int)&v87,
                  v61,
                  v62,
                  **(_QWORD **)(v9 + 40),
                  *(_QWORD *)(*(_QWORD *)(v9 + 40) + 8LL),
                  *(_OWORD *)(*(_QWORD *)(v9 + 40) + 40LL),
                  *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v9 + 40) + 80LL) + 96LL),
                  0);
          v14 = sub_33FAFB0(*a1, v63, v64, &v87, (unsigned int)v85, v86);
        }
      }
      goto LABEL_42;
    }
  }
  v78 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v9 + 40) + 80LL) + 96LL);
  v26 = sub_3400BD0(*a1, 0, (unsigned int)&v87, v85, v86, 0, 0);
  v28 = v27;
  v29 = v26;
  LODWORD(v72) = 0;
  *(_QWORD *)&v30 = sub_3400BD0(*a1, 1, (unsigned int)&v87, v85, v86, 0, v72);
  *((_QWORD *)&v70 + 1) = v28;
  *(_QWORD *)&v70 = v29;
  v14 = sub_32C7250(
          a1,
          (__int64)&v87,
          **(_QWORD **)(v9 + 40),
          *(_QWORD *)(*(_QWORD *)(v9 + 40) + 8LL),
          *(_QWORD *)(*(_QWORD *)(v9 + 40) + 40LL),
          *(_QWORD *)(*(_QWORD *)(v9 + 40) + 48LL),
          v30,
          v70,
          v78,
          1);
  if ( v14 )
  {
LABEL_42:
    v95[128] = v97[0];
    goto LABEL_6;
  }
  v95[128] = v97[0];
LABEL_44:
  v12 = sub_326B540(a2, *a1, (__int64)&v87);
  if ( v12 )
    goto LABEL_5;
  v14 = sub_32735C0(a2, a1[1], *a1, (int)&v87);
  if ( !v14 )
    v14 = 0;
LABEL_6:
  if ( v87 )
    sub_B91220((__int64)&v87, v87);
  return v14;
}
