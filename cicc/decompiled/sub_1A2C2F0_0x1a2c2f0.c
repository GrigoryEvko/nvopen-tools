// Function: sub_1A2C2F0
// Address: 0x1a2c2f0
//
void __fastcall sub_1A2C2F0(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  double v12; // xmm4_8
  double v13; // xmm5_8
  int v14; // edx
  __int64 *v15; // rax
  __int64 **v16; // r13
  _QWORD *v17; // r8
  int v18; // esi
  unsigned int v19; // ecx
  __int64 *v20; // rax
  __int64 v21; // r9
  __int64 v22; // rsi
  __int64 *v23; // r14
  __int64 v24; // rdx
  __int64 v25; // r12
  unsigned int v26; // esi
  unsigned int v27; // eax
  __int64 *v28; // rbx
  unsigned int v29; // ecx
  unsigned int v30; // r8d
  __int64 v31; // rax
  __int64 v32; // r15
  __int64 v33; // rdx
  _QWORD *v34; // r11
  __int64 *v35; // r15
  __int64 v36; // r12
  __int64 v37; // rdx
  _QWORD *v38; // r11
  _QWORD *v39; // r12
  __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // rdx
  _QWORD *v43; // rax
  _QWORD *v44; // r10
  int v45; // r11d
  _QWORD *v46; // rax
  _QWORD *v47; // rax
  __int64 v48; // [rsp+8h] [rbp-258h]
  _QWORD *v49; // [rsp+10h] [rbp-250h]
  _QWORD *v50; // [rsp+18h] [rbp-248h]
  _QWORD *v51; // [rsp+18h] [rbp-248h]
  __int64 v52; // [rsp+18h] [rbp-248h]
  __int64 v53; // [rsp+18h] [rbp-248h]
  __int64 v54; // [rsp+28h] [rbp-238h]
  __int64 v55; // [rsp+30h] [rbp-230h]
  __int64 **v56; // [rsp+70h] [rbp-1F0h]
  __int64 v57; // [rsp+88h] [rbp-1D8h] BYREF
  __m128i v58; // [rsp+90h] [rbp-1D0h] BYREF
  __int16 v59; // [rsp+A0h] [rbp-1C0h]
  __m128i v60; // [rsp+B0h] [rbp-1B0h] BYREF
  __int16 v61; // [rsp+C0h] [rbp-1A0h]
  __int64 **v62; // [rsp+D0h] [rbp-190h] BYREF
  __int64 v63; // [rsp+D8h] [rbp-188h]
  _BYTE v64[32]; // [rsp+E0h] [rbp-180h] BYREF
  __int64 v65; // [rsp+100h] [rbp-160h] BYREF
  __int64 v66; // [rsp+108h] [rbp-158h]
  _QWORD *v67; // [rsp+110h] [rbp-150h] BYREF
  unsigned int v68; // [rsp+118h] [rbp-148h]
  __int64 v69[5]; // [rsp+150h] [rbp-110h] BYREF
  int v70; // [rsp+178h] [rbp-E8h]
  __int64 v71; // [rsp+180h] [rbp-E0h]
  __int64 v72; // [rsp+188h] [rbp-D8h]
  _QWORD *v73; // [rsp+190h] [rbp-D0h]
  __int64 v74; // [rsp+198h] [rbp-C8h]
  _QWORD v75[4]; // [rsp+1A0h] [rbp-C0h] BYREF
  __int64 v76[5]; // [rsp+1C0h] [rbp-A0h] BYREF
  int v77; // [rsp+1E8h] [rbp-78h]
  __int64 v78; // [rsp+1F0h] [rbp-70h]
  __int64 v79; // [rsp+1F8h] [rbp-68h]
  _QWORD *v80; // [rsp+200h] [rbp-60h]
  __int64 v81; // [rsp+208h] [rbp-58h]
  _QWORD v82[10]; // [rsp+210h] [rbp-50h] BYREF

  v9 = sub_16498A0(a1);
  v10 = *(_QWORD *)(a1 + 48);
  LOBYTE(v75[0]) = 0;
  v69[3] = v9;
  v73 = v75;
  v11 = *(_QWORD *)(a1 + 40);
  v69[0] = 0;
  v69[1] = v11;
  v69[4] = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v74 = 0;
  v69[2] = a1 + 24;
  v76[0] = v10;
  if ( v10 )
  {
    sub_1623A60((__int64)v76, v10, 2);
    v69[0] = v76[0];
    if ( v76[0] )
      sub_1623210((__int64)v76, (unsigned __int8 *)v76[0], (__int64)v69);
  }
  v55 = *(_QWORD *)(a1 - 48);
  v54 = *(_QWORD *)(a1 - 24);
  v62 = (__int64 **)v64;
  v63 = 0x400000000LL;
  sub_1A24B00(a1, (__int64)&v62);
  v14 = v63;
  if ( (_DWORD)v63 )
  {
    v15 = (__int64 *)&v67;
    v65 = 0;
    v66 = 1;
    do
    {
      *v15 = -8;
      v15 += 2;
    }
    while ( v15 != v69 );
    v16 = v62;
    v56 = &v62[v14];
    while ( 1 )
    {
      v23 = *v16;
      v24 = **v16;
      v57 = v24;
      v25 = *(_QWORD *)*(v23 - 3);
      if ( (v66 & 1) != 0 )
      {
        v17 = &v67;
        v18 = 3;
      }
      else
      {
        v26 = v68;
        v17 = v67;
        if ( !v68 )
        {
          v27 = v66;
          ++v65;
          v28 = 0;
          v29 = ((unsigned int)v66 >> 1) + 1;
          goto LABEL_23;
        }
        v18 = v68 - 1;
      }
      v19 = v18 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v20 = &v17[2 * v19];
      v21 = *v20;
      if ( v24 != *v20 )
        break;
LABEL_17:
      v22 = v20[1];
      if ( !v22 )
      {
        v28 = v20;
LABEL_29:
        v31 = sub_16498A0(a1);
        memset(v76, 0, 24);
        v76[3] = v31;
        v80 = v82;
        v76[4] = 0;
        v77 = 0;
        v78 = 0;
        v79 = 0;
        v81 = 0;
        LOBYTE(v82[0]) = 0;
        sub_17050D0(v76, a1);
        v59 = 257;
        if ( v25 == *(_QWORD *)v55 )
        {
          v32 = v55;
        }
        else if ( *(_BYTE *)(v55 + 16) > 0x10u )
        {
          v61 = 257;
          v47 = (_QWORD *)sub_15FDF90(v55, v25, (__int64)&v60, 0);
          v32 = (__int64)sub_1A1C7B0(v76, v47, &v58);
        }
        else
        {
          v32 = sub_15A4AD0((__int64 ***)v55, v25);
        }
        v58.m128i_i64[0] = (__int64)sub_1649960(a1);
        v58.m128i_i64[1] = v33;
        v60.m128i_i64[0] = (__int64)&v58;
        v61 = 773;
        v60.m128i_i64[1] = (__int64)".sroa.speculate.load.true";
        v34 = sub_1648A60(64, 1u);
        if ( v34 )
        {
          v50 = v34;
          sub_15F9210((__int64)v34, *(_QWORD *)(*(_QWORD *)v32 + 24LL), v32, 0, 0, 0);
          v34 = v50;
        }
        v35 = sub_1A1C7B0(v76, v34, &v60);
        v59 = 257;
        if ( v25 == *(_QWORD *)v54 )
        {
          v36 = v54;
        }
        else if ( *(_BYTE *)(v54 + 16) > 0x10u )
        {
          v61 = 257;
          v46 = (_QWORD *)sub_15FDF90(v54, v25, (__int64)&v60, 0);
          v36 = (__int64)sub_1A1C7B0(v76, v46, &v58);
        }
        else
        {
          v36 = sub_15A4AD0((__int64 ***)v54, v25);
        }
        v58.m128i_i64[0] = (__int64)sub_1649960(a1);
        v61 = 773;
        v60.m128i_i64[0] = (__int64)&v58;
        v58.m128i_i64[1] = v37;
        v60.m128i_i64[1] = (__int64)".sroa.speculate.load.false";
        v38 = sub_1648A60(64, 1u);
        if ( v38 )
        {
          v51 = v38;
          sub_15F9210((__int64)v38, *(_QWORD *)(*(_QWORD *)v36 + 24LL), v36, 0, 0, 0);
          v38 = v51;
        }
        v39 = sub_1A1C7B0(v76, v38, &v60);
        sub_15F8F50((__int64)v35, 1 << (*((unsigned __int16 *)v23 + 9) >> 1) >> 1);
        sub_15F8F50((__int64)v39, 1 << (*((unsigned __int16 *)v23 + 9) >> 1) >> 1);
        if ( v23[6] || *((__int16 *)v23 + 9) < 0 )
        {
          v40 = sub_1625790((__int64)v23, 1);
          if ( v40 )
          {
            v52 = v40;
            sub_1625C10((__int64)v35, 1, v40);
            sub_1625C10((__int64)v39, 1, v52);
          }
        }
        v59 = 257;
        if ( *(_BYTE *)(*(_QWORD *)(a1 - 72) + 16LL) > 0x10u
          || *((_BYTE *)v35 + 16) > 0x10u
          || *((_BYTE *)v39 + 16) > 0x10u )
        {
          v48 = *(_QWORD *)(a1 - 72);
          v61 = 257;
          v43 = sub_1648A60(56, 3u);
          v44 = v43;
          if ( v43 )
          {
            v53 = (__int64)v43;
            v49 = v43 - 9;
            sub_15F1EA0((__int64)v43, *v35, 55, (__int64)(v43 - 9), 3, 0);
            sub_1593B40(v49, v48);
            sub_1593B40((_QWORD *)(v53 - 48), (__int64)v35);
            sub_1593B40((_QWORD *)(v53 - 24), (__int64)v39);
            sub_164B780(v53, v60.m128i_i64);
            v44 = (_QWORD *)v53;
          }
          v41 = (__int64)sub_1A1C7B0(v76, v44, &v58);
        }
        else
        {
          v41 = sub_15A2DC0(*(_QWORD *)(a1 - 72), v35, (__int64)v39, 0);
        }
        v28[1] = v41;
        v58.m128i_i64[0] = (__int64)sub_1649960((__int64)v23);
        v61 = 773;
        v58.m128i_i64[1] = v42;
        v60.m128i_i64[0] = (__int64)&v58;
        v60.m128i_i64[1] = (__int64)".sroa.speculated";
        sub_164B780(v41, v60.m128i_i64);
        if ( v80 != v82 )
          j_j___libc_free_0(v80, v82[0] + 1LL);
        if ( v76[0] )
          sub_161E7C0((__int64)v76, v76[0]);
        v22 = v28[1];
      }
      ++v16;
      sub_164D160((__int64)v23, v22, a2, a3, a4, a5, v12, v13, a8, a9);
      sub_15F20C0(v23);
      if ( v56 == v16 )
      {
        if ( (v66 & 1) == 0 )
          j___libc_free_0(v67);
        goto LABEL_5;
      }
    }
    v45 = 1;
    v28 = 0;
    while ( v21 != -8 )
    {
      if ( v21 == -16 && !v28 )
        v28 = v20;
      v19 = v18 & (v45 + v19);
      v20 = &v17[2 * v19];
      v21 = *v20;
      if ( v24 == *v20 )
        goto LABEL_17;
      ++v45;
    }
    v30 = 12;
    v26 = 4;
    if ( !v28 )
      v28 = v20;
    v27 = v66;
    ++v65;
    v29 = ((unsigned int)v66 >> 1) + 1;
    if ( (v66 & 1) == 0 )
    {
      v26 = v68;
LABEL_23:
      v30 = 3 * v26;
    }
    if ( 4 * v29 >= v30 )
    {
      sub_1A2BF40((__int64)&v65, 2 * v26);
    }
    else
    {
      if ( v26 - HIDWORD(v66) - v29 > v26 >> 3 )
      {
LABEL_26:
        LODWORD(v66) = (2 * (v27 >> 1) + 2) | v27 & 1;
        if ( *v28 != -8 )
          --HIDWORD(v66);
        *v28 = v24;
        v28[1] = 0;
        goto LABEL_29;
      }
      sub_1A2BF40((__int64)&v65, v26);
    }
    sub_1A272D0((__int64)&v65, &v57, v76);
    v28 = (__int64 *)v76[0];
    v24 = v57;
    v27 = v66;
    goto LABEL_26;
  }
LABEL_5:
  if ( v62 != (__int64 **)v64 )
    _libc_free((unsigned __int64)v62);
  if ( v73 != v75 )
    j_j___libc_free_0(v73, v75[0] + 1LL);
  if ( v69[0] )
    sub_161E7C0((__int64)v69, v69[0]);
}
