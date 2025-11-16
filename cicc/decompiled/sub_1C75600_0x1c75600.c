// Function: sub_1C75600
// Address: 0x1c75600
//
__int64 __fastcall sub_1C75600(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 v14; // rbx
  __int64 v16; // r13
  __int64 v17; // rbx
  _QWORD *v18; // r15
  _QWORD *v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 *v22; // r14
  __int64 v23; // r12
  _QWORD *v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // r13
  __int64 v29; // r14
  _QWORD *v30; // rax
  _QWORD *v31; // rax
  __int64 v32; // rsi
  __int64 v33; // r9
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // r14
  __int64 v40; // r12
  int v41; // eax
  __int64 v42; // rax
  int v43; // edx
  __int64 v44; // rdx
  __int64 *v45; // rax
  __int64 v46; // rcx
  unsigned __int64 v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rax
  __int64 v52; // rax
  double v53; // xmm4_8
  double v54; // xmm5_8
  __int64 v56; // r12
  _QWORD *v57; // rax
  __int64 v58; // rax
  double v59; // xmm4_8
  double v60; // xmm5_8
  _QWORD *v61; // rdx
  __int64 v62; // [rsp+0h] [rbp-C0h]
  __int64 v63; // [rsp+8h] [rbp-B8h]
  __int64 v64; // [rsp+10h] [rbp-B0h]
  __int64 v66; // [rsp+20h] [rbp-A0h]
  __int64 *v68; // [rsp+30h] [rbp-90h]
  int v70; // [rsp+40h] [rbp-80h]
  __int64 v71; // [rsp+48h] [rbp-78h]
  __int64 v72; // [rsp+48h] [rbp-78h]
  __int64 v73; // [rsp+50h] [rbp-70h]
  __int64 v74; // [rsp+58h] [rbp-68h] BYREF
  __int64 v75; // [rsp+68h] [rbp-58h] BYREF
  __int64 v76[2]; // [rsp+70h] [rbp-50h] BYREF
  char v77; // [rsp+80h] [rbp-40h]
  char v78; // [rsp+81h] [rbp-3Fh]

  v14 = *(_QWORD *)(a3 + 48);
  v74 = a1;
  v66 = a3 + 40;
  while ( v66 != v14 )
  {
    if ( !v14 )
      BUG();
    if ( *(_BYTE *)(v14 - 8) != 77 )
      return 1;
    v64 = v14 - 24;
    v73 = sub_1C741B0(v14 - 24, a4);
    if ( *(_BYTE *)(v73 + 16) <= 0x17u )
      return 0;
    v16 = v74;
    if ( !*(_QWORD *)(v73 + 8) )
      goto LABEL_38;
    v63 = v14;
    v17 = *(_QWORD *)(v73 + 8);
    v70 = 0;
    v71 = v74 + 56;
    v62 = a6;
    do
    {
      v21 = sub_1648700(v17);
      v22 = v21;
      if ( *((_BYTE *)v21 + 16) <= 0x17u )
        goto LABEL_12;
      v23 = v21[5];
      v24 = *(_QWORD **)(v16 + 72);
      v19 = *(_QWORD **)(v16 + 64);
      if ( v24 == v19 )
      {
        v18 = &v19[*(unsigned int *)(v16 + 84)];
        if ( v19 == v18 )
        {
          v61 = *(_QWORD **)(v16 + 64);
        }
        else
        {
          do
          {
            if ( v23 == *v19 )
              break;
            ++v19;
          }
          while ( v18 != v19 );
          v61 = v18;
        }
      }
      else
      {
        v18 = &v24[*(unsigned int *)(v16 + 80)];
        v19 = sub_16CC9F0(v71, v23);
        if ( v23 == *v19 )
        {
          v25 = *(_QWORD *)(v16 + 72);
          if ( v25 == *(_QWORD *)(v16 + 64) )
            v26 = *(unsigned int *)(v16 + 84);
          else
            v26 = *(unsigned int *)(v16 + 80);
          v61 = (_QWORD *)(v25 + 8 * v26);
        }
        else
        {
          v20 = *(_QWORD *)(v16 + 72);
          if ( v20 != *(_QWORD *)(v16 + 64) )
          {
            v19 = (_QWORD *)(v20 + 8LL * *(unsigned int *)(v16 + 80));
            goto LABEL_10;
          }
          v19 = (_QWORD *)(v20 + 8LL * *(unsigned int *)(v16 + 84));
          v61 = v19;
        }
      }
      while ( v61 != v19 && *v19 >= 0xFFFFFFFFFFFFFFFELL )
        ++v19;
LABEL_10:
      if ( v19 == v18 )
      {
        ++v70;
        v68 = v22;
      }
LABEL_12:
      v17 = *(_QWORD *)(v17 + 8);
    }
    while ( v17 );
    v14 = v63;
    a6 = v62;
    if ( v70 != 1 )
      goto LABEL_30;
    v27 = v68[5];
    if ( v27 == v62 )
      goto LABEL_54;
    if ( v27 != a5 )
    {
LABEL_30:
      v28 = *(_QWORD *)(v73 + 8);
      if ( v28 )
      {
        v29 = *(_QWORD *)(v73 + 8);
        while ( 1 )
        {
          v30 = sub_1648700(v29);
          if ( *((_BYTE *)v30 + 16) > 0x17u && a5 == v30[5] )
            return 0;
          v29 = *(_QWORD *)(v29 + 8);
          if ( !v29 )
          {
            while ( 1 )
            {
              v31 = sub_1648700(v28);
              if ( *((_BYTE *)v31 + 16) > 0x17u && v62 == v31[5] )
                return 0;
              v28 = *(_QWORD *)(v28 + 8);
              if ( !v28 )
                goto LABEL_38;
            }
          }
        }
      }
LABEL_38:
      v32 = *(_QWORD *)v73;
      v78 = 1;
      v77 = 3;
      v33 = *(_QWORD *)(a5 + 48);
      if ( v33 )
        v33 -= 24;
      v76[0] = (__int64)"splitPhi";
      v72 = v33;
      v34 = sub_1648B60(64);
      v38 = v72;
      v39 = v34;
      if ( v34 )
      {
        sub_15F1EA0(v34, v32, 53, 0, 0, v72);
        v40 = v39;
        *(_DWORD *)(v39 + 56) = 1;
        sub_164B780(v39, v76);
        v32 = *(unsigned int *)(v39 + 56);
        sub_1648880(v39, v32, 1);
      }
      else
      {
        v40 = 0;
      }
      v41 = *(_DWORD *)(v39 + 20) & 0xFFFFFFF;
      if ( v41 == *(_DWORD *)(v39 + 56) )
      {
        sub_15F55D0(v39, v32, v35, v36, v37, v38);
        v41 = *(_DWORD *)(v39 + 20) & 0xFFFFFFF;
      }
      v42 = (v41 + 1) & 0xFFFFFFF;
      v43 = v42 | *(_DWORD *)(v39 + 20) & 0xF0000000;
      *(_DWORD *)(v39 + 20) = v43;
      if ( (v43 & 0x40000000) != 0 )
        v44 = *(_QWORD *)(v39 - 8);
      else
        v44 = v40 - 24 * v42;
      v45 = (__int64 *)(v44 + 24LL * (unsigned int)(v42 - 1));
      if ( *v45 )
      {
        v46 = v45[1];
        v47 = v45[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v47 = v46;
        if ( v46 )
          *(_QWORD *)(v46 + 16) = *(_QWORD *)(v46 + 16) & 3LL | v47;
      }
      *v45 = v73;
      v48 = *(_QWORD *)(v73 + 8);
      v45[1] = v48;
      if ( v48 )
        *(_QWORD *)(v48 + 16) = (unsigned __int64)(v45 + 1) | *(_QWORD *)(v48 + 16) & 3LL;
      v45[2] = (v73 + 8) | v45[2] & 3;
      *(_QWORD *)(v73 + 8) = v45;
      v49 = *(_DWORD *)(v39 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v39 + 23) & 0x40) != 0 )
        v50 = *(_QWORD *)(v39 - 8);
      else
        v50 = v40 - 24 * v49;
      *(_QWORD *)(v50 + 8LL * (unsigned int)(v49 - 1) + 24LL * *(unsigned int *)(v39 + 56) + 8) = a4;
      v75 = v39;
      v51 = sub_1C74150(v64, a4);
      v52 = sub_1C74800(a6, *(_QWORD *)v39, v39, a5, v51, a2);
      v76[0] = (__int64)&v75;
      v76[1] = (__int64)&v74;
      sub_164C7D0(
        v73,
        v52,
        (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_1C74290,
        (__int64)v76,
        a7,
        a8,
        a9,
        a10,
        v53,
        v54,
        a13,
        a14);
      goto LABEL_54;
    }
    v56 = v68[1];
    if ( !v56 )
    {
LABEL_69:
      v58 = sub_1C74150(v64, a4);
      v75 = sub_1C74800(v62, *v68, (__int64)v68, a5, v58, a2);
      v76[0] = (__int64)&v75;
      sub_164C7D0(
        (__int64)v68,
        v75,
        (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_1C73B70,
        (__int64)v76,
        a7,
        a8,
        a9,
        a10,
        v59,
        v60,
        a13,
        a14);
      goto LABEL_54;
    }
    v57 = sub_1648700(v68[1]);
    if ( *(_QWORD *)(v56 + 8) || !v57 || v57[5] != v62 )
    {
      while ( *((_BYTE *)v57 + 16) <= 0x17u || v62 != v57[5] )
      {
        v56 = *(_QWORD *)(v56 + 8);
        if ( !v56 )
          goto LABEL_69;
        v57 = sub_1648700(v56);
      }
      return 0;
    }
LABEL_54:
    v14 = *(_QWORD *)(v14 + 8);
  }
  return 1;
}
