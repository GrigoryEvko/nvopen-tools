// Function: sub_387B3F0
// Address: 0x387b3f0
//
__int64 ***__fastcall sub_387B3F0(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 ***v11; // r15
  bool v12; // sf
  int v13; // eax
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 ***v16; // r12
  double v17; // xmm4_8
  double v18; // xmm5_8
  double v19; // xmm4_8
  double v20; // xmm5_8
  __int64 ***v21; // rax
  __int64 v22; // r14
  __int64 v23; // r15
  __int64 v24; // rax
  _QWORD *v25; // rax
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  __int64 **v31; // rax
  __int64 v32; // rdx
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 *v36; // r12
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rsi
  __int64 v40; // rsi
  unsigned __int8 *v41; // rsi
  _QWORD *v42; // rax
  __int64 v43; // rbx
  __int64 **v44; // rax
  __int64 *v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rdi
  __int64 v48; // rsi
  __int64 v49; // rax
  __int64 v50; // rsi
  __int64 v51; // rsi
  __int64 v52; // rdx
  unsigned __int8 *v53; // rsi
  __int64 **v54; // rax
  double v55; // xmm4_8
  double v56; // xmm5_8
  _QWORD *v58; // [rsp+10h] [rbp-B0h]
  __int64 v59; // [rsp+18h] [rbp-A8h]
  __int64 *v60; // [rsp+18h] [rbp-A8h]
  __int64 *v61; // [rsp+18h] [rbp-A8h]
  __int64 **v63; // [rsp+28h] [rbp-98h]
  __int64 v64; // [rsp+30h] [rbp-90h]
  __int64 v65; // [rsp+38h] [rbp-88h]
  int v66; // [rsp+38h] [rbp-88h]
  unsigned __int8 *v67; // [rsp+48h] [rbp-78h] BYREF
  __int64 v68[2]; // [rsp+50h] [rbp-70h] BYREF
  __int16 v69; // [rsp+60h] [rbp-60h]
  __int64 v70[2]; // [rsp+70h] [rbp-50h] BYREF
  __int16 v71; // [rsp+80h] [rbp-40h]

  v11 = (__int64 ***)sub_3875200(
                       a1,
                       *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL * ((unsigned int)*(_QWORD *)(a2 + 40) - 1)),
                       *(double *)a3.m128_u64,
                       a4,
                       a5);
  v63 = *v11;
  v65 = *(_QWORD *)(a2 + 40);
  v12 = (int)v65 - 2 < 0;
  v13 = v65 - 2;
  v66 = v65 - 2;
  if ( !v12 )
  {
    v14 = (__int64)v11;
    v64 = 8LL * v13;
    do
    {
      v16 = (__int64 ***)v14;
      if ( v63 != (__int64 **)sub_1456040(*(_QWORD *)(*(_QWORD *)(a2 + 32) + v64)) )
      {
        v63 = (__int64 **)sub_1456E10(*a1, (__int64)v63);
        v16 = sub_38744E0(a1, v14, v63, a3, a4, a5, a6, v19, v20, a9, a10);
      }
      v21 = sub_38761C0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + v64), v63, a3, a4, a5, a6, v17, v18, a9, a10);
      v69 = 257;
      v22 = (__int64)v21;
      if ( *((_BYTE *)v16 + 16) > 0x10u || *((_BYTE *)v21 + 16) > 0x10u )
      {
        v71 = 257;
        v42 = sub_1648A60(56, 2u);
        v23 = (__int64)v42;
        if ( v42 )
        {
          v43 = (__int64)v42;
          v44 = *v16;
          if ( *((_BYTE *)*v16 + 8) == 16 )
          {
            v60 = v44[4];
            v45 = (__int64 *)sub_1643320(*v44);
            v46 = (__int64)sub_16463B0(v45, (unsigned int)v60);
          }
          else
          {
            v46 = sub_1643320(*v44);
          }
          sub_15FEC10(v23, v46, 51, 38, (__int64)v16, v22, (__int64)v70, 0);
        }
        else
        {
          v43 = 0;
        }
        v47 = a1[34];
        if ( v47 )
        {
          v61 = (__int64 *)a1[35];
          sub_157E9D0(v47 + 40, v23);
          v48 = *v61;
          v49 = *(_QWORD *)(v23 + 24) & 7LL;
          *(_QWORD *)(v23 + 32) = v61;
          v48 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v23 + 24) = v48 | v49;
          *(_QWORD *)(v48 + 8) = v23 + 24;
          *v61 = *v61 & 7 | (v23 + 24);
        }
        sub_164B780(v43, v68);
        v50 = a1[33];
        if ( v50 )
        {
          v67 = (unsigned __int8 *)a1[33];
          sub_1623A60((__int64)&v67, v50, 2);
          v51 = *(_QWORD *)(v23 + 48);
          v52 = v23 + 48;
          if ( v51 )
          {
            sub_161E7C0(v23 + 48, v51);
            v52 = v23 + 48;
          }
          v53 = v67;
          *(_QWORD *)(v23 + 48) = v67;
          if ( v53 )
            sub_1623210((__int64)&v67, v53, v52);
        }
      }
      else
      {
        v23 = sub_15A37B0(0x26u, v16, v21, 0);
        v24 = sub_14DBA30(v23, a1[41], 0);
        if ( v24 )
          v23 = v24;
      }
      sub_38740E0((__int64)a1, v23);
      v68[0] = (__int64)"smax";
      v69 = 259;
      if ( *(_BYTE *)(v23 + 16) > 0x10u || *((_BYTE *)v16 + 16) > 0x10u || *(_BYTE *)(v22 + 16) > 0x10u )
      {
        v71 = 257;
        v25 = sub_1648A60(56, 3u);
        v14 = (__int64)v25;
        if ( v25 )
        {
          v58 = v25 - 9;
          v59 = (__int64)v25;
          sub_15F1EA0((__int64)v25, (__int64)*v16, 55, (__int64)(v25 - 9), 3, 0);
          if ( *(_QWORD *)(v14 - 72) )
          {
            v26 = *(_QWORD *)(v14 - 64);
            v27 = *(_QWORD *)(v14 - 56) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v27 = v26;
            if ( v26 )
              *(_QWORD *)(v26 + 16) = *(_QWORD *)(v26 + 16) & 3LL | v27;
          }
          *(_QWORD *)(v14 - 72) = v23;
          v28 = *(_QWORD *)(v23 + 8);
          *(_QWORD *)(v14 - 64) = v28;
          if ( v28 )
            *(_QWORD *)(v28 + 16) = (v14 - 64) | *(_QWORD *)(v28 + 16) & 3LL;
          *(_QWORD *)(v14 - 56) = (v23 + 8) | *(_QWORD *)(v14 - 56) & 3LL;
          *(_QWORD *)(v23 + 8) = v58;
          if ( *(_QWORD *)(v14 - 48) )
          {
            v29 = *(_QWORD *)(v14 - 40);
            v30 = *(_QWORD *)(v14 - 32) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v30 = v29;
            if ( v29 )
              *(_QWORD *)(v29 + 16) = *(_QWORD *)(v29 + 16) & 3LL | v30;
          }
          *(_QWORD *)(v14 - 48) = v16;
          v31 = v16[1];
          *(_QWORD *)(v14 - 40) = v31;
          if ( v31 )
            v31[2] = (__int64 *)((v14 - 40) | (unsigned __int64)v31[2] & 3);
          *(_QWORD *)(v14 - 32) = (unsigned __int64)(v16 + 1) | *(_QWORD *)(v14 - 32) & 3LL;
          v16[1] = (__int64 **)(v14 - 48);
          if ( *(_QWORD *)(v14 - 24) )
          {
            v32 = *(_QWORD *)(v14 - 16);
            v33 = *(_QWORD *)(v14 - 8) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v33 = v32;
            if ( v32 )
              *(_QWORD *)(v32 + 16) = *(_QWORD *)(v32 + 16) & 3LL | v33;
          }
          *(_QWORD *)(v14 - 24) = v22;
          if ( v22 )
          {
            v34 = *(_QWORD *)(v22 + 8);
            *(_QWORD *)(v14 - 16) = v34;
            if ( v34 )
              *(_QWORD *)(v34 + 16) = (v14 - 16) | *(_QWORD *)(v34 + 16) & 3LL;
            *(_QWORD *)(v14 - 8) = (v22 + 8) | *(_QWORD *)(v14 - 8) & 3LL;
            *(_QWORD *)(v22 + 8) = v14 - 24;
          }
          sub_164B780(v14, v70);
        }
        else
        {
          v59 = 0;
        }
        v35 = a1[34];
        if ( v35 )
        {
          v36 = (__int64 *)a1[35];
          sub_157E9D0(v35 + 40, v14);
          v37 = *(_QWORD *)(v14 + 24);
          v38 = *v36;
          *(_QWORD *)(v14 + 32) = v36;
          v38 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v14 + 24) = v38 | v37 & 7;
          *(_QWORD *)(v38 + 8) = v14 + 24;
          *v36 = *v36 & 7 | (v14 + 24);
        }
        sub_164B780(v59, v68);
        v39 = a1[33];
        if ( v39 )
        {
          v70[0] = a1[33];
          sub_1623A60((__int64)v70, v39, 2);
          v40 = *(_QWORD *)(v14 + 48);
          if ( v40 )
            sub_161E7C0(v14 + 48, v40);
          v41 = (unsigned __int8 *)v70[0];
          *(_QWORD *)(v14 + 48) = v70[0];
          if ( v41 )
            sub_1623210((__int64)v70, v41, v14 + 48);
        }
      }
      else
      {
        v14 = sub_15A2DC0(v23, (__int64 *)v16, v22, 0);
        v15 = sub_14DBA30(v14, a1[41], 0);
        if ( v15 )
          v14 = v15;
      }
      sub_38740E0((__int64)a1, v14);
      --v66;
      v64 -= 8;
    }
    while ( v66 != -1 );
    v11 = (__int64 ***)v14;
    v63 = *(__int64 ***)v14;
  }
  if ( v63 == (__int64 **)sub_1456040(**(_QWORD **)(a2 + 32)) )
    return v11;
  v54 = (__int64 **)sub_1456040(**(_QWORD **)(a2 + 32));
  return sub_38744E0(a1, (__int64)v11, v54, a3, a4, a5, a6, v55, v56, a9, a10);
}
