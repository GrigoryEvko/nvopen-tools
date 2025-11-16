// Function: sub_8C18E0
// Address: 0x8c18e0
//
_QWORD *__fastcall sub_8C18E0(__m128i *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  unsigned __int8 v6; // al
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __m128i *v12; // rax
  __m128i *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 *v16; // r9
  __int64 v17; // rax
  __int64 v18; // r13
  __m128i *v19; // rax
  __int64 v20; // r13
  __int64 v21; // rdx
  _QWORD *v22; // r14
  __int64 *v23; // rbx
  __m128i *v24; // r12
  __int64 **v25; // rax
  __int64 **v26; // r11
  __m128i *v27; // r10
  __int64 v28; // r11
  _QWORD *v29; // rax
  __m128i *v30; // rax
  unsigned int v32; // eax
  __int64 v33; // r14
  __m128i *v34; // rax
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 **v38; // rax
  __int64 **v39; // rax
  __m128i *v40; // rax
  __int64 v41; // [rsp-10h] [rbp-160h]
  __int64 v42; // [rsp-10h] [rbp-160h]
  __m128i *v43; // [rsp-10h] [rbp-160h]
  __m128i *v44; // [rsp-8h] [rbp-158h]
  __int64 v45; // [rsp+8h] [rbp-148h]
  __int64 *v46; // [rsp+8h] [rbp-148h]
  __m128i *v47; // [rsp+10h] [rbp-140h]
  __int64 v48; // [rsp+18h] [rbp-138h]
  __int64 v49; // [rsp+18h] [rbp-138h]
  int v50; // [rsp+24h] [rbp-12Ch]
  __int64 v51; // [rsp+28h] [rbp-128h]
  __int64 v52; // [rsp+28h] [rbp-128h]
  _QWORD *v53; // [rsp+30h] [rbp-120h]
  __int64 v54; // [rsp+38h] [rbp-118h]
  unsigned __int64 v55; // [rsp+38h] [rbp-118h]
  _QWORD *v57; // [rsp+48h] [rbp-108h]
  __m128i *v58; // [rsp+50h] [rbp-100h]
  _QWORD *v59; // [rsp+58h] [rbp-F8h]
  __int64 *v60; // [rsp+58h] [rbp-F8h]
  __m128i *v61; // [rsp+58h] [rbp-F8h]
  __int64 **v62; // [rsp+58h] [rbp-F8h]
  unsigned int v63; // [rsp+6Ch] [rbp-E4h] BYREF
  __int64 *v64; // [rsp+70h] [rbp-E0h] BYREF
  __m128i *v65; // [rsp+78h] [rbp-D8h] BYREF
  __m128i *v66; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v67; // [rsp+88h] [rbp-C8h]
  __int64 v68; // [rsp+90h] [rbp-C0h]
  __m128i v69; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v70; // [rsp+B0h] [rbp-A0h]
  __m128i v71; // [rsp+C0h] [rbp-90h] BYREF
  _QWORD *v72; // [rsp+D0h] [rbp-80h]
  _QWORD *v73; // [rsp+D8h] [rbp-78h]
  __int64 *v74; // [rsp+E0h] [rbp-70h]
  __int64 v75; // [rsp+E8h] [rbp-68h]
  __int64 v76; // [rsp+F0h] [rbp-60h]

  v5 = (__int64)a1;
  v6 = *(_BYTE *)(a3 + 80);
  v7 = a1[5].m128i_i64[1];
  v51 = a2;
  v64 = 0;
  v63 = 0;
  if ( v6 == 10 )
  {
    v53 = *(_QWORD **)(a3 + 88);
    sub_865900((__int64)a1);
    v50 = 0;
    v48 = 0;
    goto LABEL_9;
  }
  if ( v6 == 9 )
  {
    v48 = *(_QWORD *)(*(_QWORD *)(a3 + 96) + 56LL);
    goto LABEL_7;
  }
  if ( v6 > 9u )
  {
    if ( (unsigned __int8)(v6 - 19) <= 3u )
    {
      v48 = *(_QWORD *)(a3 + 88);
      goto LABEL_7;
    }
LABEL_26:
    BUG();
  }
  if ( v6 <= 5u )
  {
    if ( v6 > 3u )
    {
      v48 = *(_QWORD *)(*(_QWORD *)(a3 + 96) + 80LL);
      goto LABEL_7;
    }
    goto LABEL_26;
  }
  if ( v6 != 6 )
    goto LABEL_26;
  v48 = *(_QWORD *)(*(_QWORD *)(a3 + 96) + 32LL);
LABEL_7:
  v53 = *(_QWORD **)(v48 + 176);
  sub_865900((__int64)a1);
  if ( (*(_BYTE *)(v48 + 160) & 4) != 0 )
    goto LABEL_31;
  v50 = 1;
LABEL_9:
  v57 = sub_8907A0((__int64)a1, a2, a3, a3, a4);
  v57[6] = *(_QWORD *)(a3 + 48);
  v59 = **(_QWORD ***)(v7 + 32);
  sub_892DC0((__int64)v59, &v64, &v65, 0, 1);
  v12 = (__m128i *)sub_896D70((__int64)a1, (__int64)v64, 1);
  v66 = 0;
  v58 = v12;
  v67 = 0;
  v68 = 0;
  v13 = (__m128i *)sub_823970(24);
  v67 = 1;
  v66 = v13;
  v17 = 1;
  v70 = 0;
  if ( a1 != (__m128i *)a4 )
  {
    sub_89F970(*(_QWORD *)(a4 + 64), (__int64)&v66);
    v17 = v67;
  }
  v18 = v68;
  if ( v68 == v17 )
    sub_738390((const __m128i **)&v66);
  v19 = (__m128i *)((char *)v66 + 24 * v18);
  if ( v19 )
  {
    v14 = (__int64)v59;
    v69.m128i_i64[1] = (__int64)v58;
    v69.m128i_i64[0] = (__int64)v59;
    *v19 = _mm_loadu_si128(&v69);
    v19[1].m128i_i64[0] = v70;
  }
  v20 = v18 + 1;
  v21 = v63;
  v68 = v20;
  if ( (_DWORD)v20 )
  {
    v46 = v64;
    sub_892150(&v71);
    v21 = v63;
    v32 = v20 - 1;
    if ( (int)v20 - 1 >= 0 )
    {
      v21 = (int)v32;
      v16 = (__int64 *)v63;
      v33 = 24LL * (int)v32;
      v55 = 24 * ((int)v20 - (unsigned __int64)v32) - 48;
      v34 = v66;
      a1 = v66;
      if ( !v63 )
      {
        v35 = v33;
        while ( 1 )
        {
          sub_8C0720(
            v5,
            (__int64)v46,
            *(_QWORD **)((char *)v34->m128i_i64 + v35),
            *(__m128i **)((char *)&v34->m128i_i64[1] + v35),
            24576,
            (int *)&v63,
            &v71);
          v15 = v63;
          if ( v63 )
          {
            a1 = v66;
            goto LABEL_41;
          }
          v35 -= 24;
          if ( v55 == v35 )
            break;
          v34 = v66;
        }
        a1 = v66;
        a2 = 24 * v67;
        goto LABEL_17;
      }
LABEL_41:
      a2 = 24 * v67;
LABEL_42:
      sub_823A00((__int64)a1, a2, v21, v14, v15, v16);
      v57 = 0;
      goto LABEL_32;
    }
  }
  a1 = v66;
  a2 = 24 * v67;
  if ( (_DWORD)v21 )
    goto LABEL_42;
LABEL_17:
  sub_823A00((__int64)a1, a2, v21, v14, v15, v16);
  sub_892150(&v71);
  v72 = v59;
  v69.m128i_i64[0] = (__int64)sub_72F240(v58);
  v54 = sub_8A0370(a4, (__m128i **)&v69, 0, 0, 0, 0, 0)[11];
  if ( v50 )
  {
    v73 = **(_QWORD ***)(v48 + 328);
    v22 = v73;
    sub_892DC0((__int64)v73, &v64, &v66, 0, 0);
    v30 = (__m128i *)sub_896D70(v5, (__int64)v66, 1);
    a2 = (__int64)v64;
    a1 = (__m128i *)a3;
    v47 = v30;
    v74 = v64;
    sub_8C0720(a3, (__int64)v64, v22, v30, 24576, (int *)&v63, &v71);
    v11 = v41;
    if ( v63 )
      goto LABEL_31;
    sub_8C0720(a3, (__int64)v66, v59, v58, 24576, (int *)&v63, &v71);
    v10 = v63;
    a2 = v42;
    a1 = v44;
    if ( v63 )
      goto LABEL_31;
  }
  else
  {
    v47 = 0;
    v22 = 0;
  }
  v23 = (__int64 *)(v5 + 48);
  v49 = v57[11];
  **(_QWORD **)(v49 + 32) = v64;
  v24 = sub_725FD0();
  a2 = (__int64)v58;
  v74 = v64;
  v75 = v51;
  v76 = v54;
  a1 = (__m128i *)v53[19];
  v25 = sub_8A2270((__int64)a1, v58, (__int64)v59, v23, 24576, (int *)&v63, &v71);
  v9 = v63;
  v26 = v25;
  v8 = (__int64)v44;
  if ( !v63 )
  {
    v52 = v53[27];
    if ( v52 )
    {
      a2 = (__int64)v58;
      v45 = (__int64)v25;
      a1 = *(__m128i **)v52;
      v27 = (__m128i *)sub_743530(*(__m128i **)v52, v58, (__int64)v59, 24580, (int *)&v63, v71.m128i_i64);
      if ( v63 )
        goto LABEL_31;
      v28 = v45;
      if ( !v50 )
        goto LABEL_23;
      a2 = (__int64)v47;
      v61 = v27;
      v39 = sub_8A2270(v45, v47, (__int64)v22, v23, 24576, (int *)&v63, &v71);
      v11 = v63;
      a1 = v43;
      v26 = v39;
      v10 = (__int64)v44;
      if ( v63 )
        goto LABEL_31;
      if ( v61 )
      {
        a1 = v61;
        v62 = v39;
        v40 = (__m128i *)sub_743530(a1, v47, (__int64)v22, 24580, (int *)&v63, v71.m128i_i64);
        a2 = v63;
        v28 = (__int64)v62;
        v27 = v40;
        if ( !v63 )
        {
LABEL_23:
          v60 = (__int64 *)v27;
          *(_QWORD *)(v28 + 160) = v54;
          v24[9].m128i_i64[1] = v28;
          if ( v27 )
          {
            v29 = sub_7272D0();
            v24[13].m128i_i64[1] = (__int64)v29;
            *v29 = v60;
            v8 = *(_QWORD *)(v52 + 8);
            *(_QWORD *)(v24[13].m128i_i64[1] + 8) = v8;
          }
          goto LABEL_45;
        }
        goto LABEL_31;
      }
LABEL_44:
      v26[20] = (__int64 *)v54;
      v24[9].m128i_i64[1] = (__int64)v26;
LABEL_45:
      *(_QWORD *)(v49 + 176) = v24;
      sub_877F10((__int64)v24, a4, v8, v9, v10, v11);
      sub_725ED0((__int64)v24, 7);
      a2 = 0xFFFFFFFFLL;
      a1 = v24;
      v36 = *(_QWORD *)(*(_QWORD *)(a4 + 88) + 104LL);
      v24[12].m128i_i8[1] |= 0x10u;
      v24[11].m128i_i64[0] = v36;
      v37 = v53[8];
      v24[12].m128i_i8[3] |= 8u;
      v24[4].m128i_i64[0] = v37;
      sub_7362F0((__int64)v24, -1);
      *(_QWORD *)(*(_QWORD *)(v49 + 104) + 192LL) = v24;
      goto LABEL_32;
    }
    if ( !v50 )
      goto LABEL_44;
    a1 = (__m128i *)v25;
    a2 = (__int64)v47;
    v38 = sub_8A2270((__int64)v25, v47, (__int64)v22, v23, 24576, (int *)&v63, &v71);
    v9 = v63;
    v26 = v38;
    v8 = (__int64)v44;
    if ( !v63 )
      goto LABEL_44;
  }
LABEL_31:
  v57 = 0;
LABEL_32:
  sub_864110((__int64)a1, a2, v8, v9, v10, (__int64 *)v11);
  return v57;
}
