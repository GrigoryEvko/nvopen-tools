// Function: sub_1A2FFA0
// Address: 0x1a2ffa0
//
bool __fastcall sub_1A2FFA0(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 *v6; // r13
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // r8
  unsigned __int64 v11; // rcx
  unsigned int v12; // r12d
  unsigned __int64 v13; // rax
  __int64 *v14; // rax
  __int64 *v15; // rax
  __int64 v16; // r9
  __int64 v17; // rsi
  unsigned int v18; // r12d
  _QWORD *v19; // r14
  __int64 v20; // rdi
  unsigned int v21; // r12d
  bool v22; // al
  __int64 v23; // rdx
  unsigned int v24; // r12d
  _QWORD *v25; // r13
  __int64 v26; // rdi
  unsigned int v27; // ebx
  bool result; // al
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned int v31; // eax
  __int64 v32; // rsi
  __int64 v33; // r12
  __int64 *v34; // rax
  __int64 *v35; // r10
  __int64 v36; // rax
  __int64 v37; // rsi
  __int64 *v38; // rax
  unsigned int v39; // r8d
  __int64 *v40; // rax
  _BYTE *v41; // rdi
  __int64 v42; // r8
  unsigned __int64 v43; // r8
  _QWORD *v44; // rdi
  __int64 **v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 **v48; // rax
  __int64 v49; // rax
  _BYTE *v50; // r12
  __int64 *v51; // rax
  __int64 v52; // rax
  __int64 v53; // r13
  __int64 *v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rsi
  unsigned __int8 *v57; // rax
  unsigned __int8 *v58; // rcx
  unsigned __int64 v59; // rax
  __int64 ***v60; // rax
  _QWORD *v61; // rdi
  __int64 v62; // rax
  __int64 **v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 **v66; // rax
  __int64 v67; // rax
  __int64 *v68; // r12
  unsigned int v69; // r14d
  __int64 v70; // rbx
  _QWORD *v71; // rax
  _QWORD *v72; // rdi
  __int64 v73; // [rsp+18h] [rbp-D8h]
  __int64 v74; // [rsp+18h] [rbp-D8h]
  __int64 v75; // [rsp+20h] [rbp-D0h]
  __int64 **v76; // [rsp+20h] [rbp-D0h]
  __int64 v77; // [rsp+20h] [rbp-D0h]
  __int64 **v78; // [rsp+20h] [rbp-D0h]
  __int64 v79; // [rsp+20h] [rbp-D0h]
  __int64 *v80; // [rsp+28h] [rbp-C8h]
  __int64 *v81; // [rsp+28h] [rbp-C8h]
  __int64 *v82; // [rsp+28h] [rbp-C8h]
  __int64 v83; // [rsp+28h] [rbp-C8h]
  __int64 v84; // [rsp+28h] [rbp-C8h]
  unsigned int v85; // [rsp+28h] [rbp-C8h]
  _BYTE *v86; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v87; // [rsp+30h] [rbp-C0h]
  unsigned int v88; // [rsp+30h] [rbp-C0h]
  __int64 *v89; // [rsp+30h] [rbp-C0h]
  __int64 v90; // [rsp+30h] [rbp-C0h]
  __int64 v91; // [rsp+30h] [rbp-C0h]
  __int64 *v92; // [rsp+30h] [rbp-C0h]
  unsigned __int8 v93; // [rsp+30h] [rbp-C0h]
  __int64 v94; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v95; // [rsp+48h] [rbp-A8h]
  __int64 v96; // [rsp+50h] [rbp-A0h]
  __m128i v97; // [rsp+60h] [rbp-90h] BYREF
  char v98; // [rsp+70h] [rbp-80h]
  char v99; // [rsp+71h] [rbp-7Fh]
  __m128i v100[2]; // [rsp+80h] [rbp-70h] BYREF
  __m128i v101; // [rsp+A0h] [rbp-50h] BYREF
  char v102; // [rsp+B0h] [rbp-40h]
  char v103; // [rsp+B1h] [rbp-3Fh]

  v6 = (__int64 *)(a1 + 192);
  v94 = 0;
  v95 = 0;
  v96 = 0;
  sub_14A8180(a2, &v94, 0);
  if ( *(_BYTE *)(*(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) + 16LL) > 0x10u )
  {
    v29 = sub_1A246E0((__int64 *)a1, (__int64)v6, **(_QWORD **)(a1 + 168));
    sub_1593B40((_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), v29);
    v30 = *(_QWORD *)(a1 + 48);
    v31 = (unsigned int)(1 << *(_WORD *)(v30 + 18)) >> 1;
    if ( !v31 )
      v31 = sub_15A9FE0(*(_QWORD *)a1, *(_QWORD *)(v30 + 56));
    v32 = *(_QWORD *)(a1 + 128) - *(_QWORD *)(a1 + 56);
    v33 = (v32 | v31) & -(v32 | v31);
    v101.m128i_i64[0] = *(_QWORD *)(a2 + 56);
    v34 = (__int64 *)sub_16498A0(a2);
    *(_QWORD *)(a2 + 56) = sub_1563C10(v101.m128i_i64, v34, 1, 1);
    if ( (_DWORD)v33 )
    {
      v51 = (__int64 *)sub_16498A0(a2);
      v52 = sub_155D330(v51, (unsigned int)v33);
      v100[0].m128i_i32[0] = 0;
      v53 = v52;
      v101.m128i_i64[0] = *(_QWORD *)(a2 + 56);
      v54 = (__int64 *)sub_16498A0(a2);
      *(_QWORD *)(a2 + 56) = sub_1563E10(v101.m128i_i64, v54, v100[0].m128i_i32, 1, v53);
    }
    v101.m128i_i64[0] = *(_QWORD *)(a1 + 168);
    if ( (unsigned __int8)sub_1AE9990(v101.m128i_i64[0], 0) )
      sub_1A2EDE0(*(_QWORD *)(a1 + 32) + 208LL, v101.m128i_i64);
    return 0;
  }
  else
  {
    v8 = *(_QWORD *)(a1 + 32);
    v101.m128i_i64[0] = a2;
    sub_1A2EDE0(v8 + 208, v101.m128i_i64);
    v9 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 56LL);
    v10 = v9;
    if ( *(_BYTE *)(v9 + 8) == 16 )
    {
      v10 = **(_QWORD **)(v9 + 16);
      if ( *(_QWORD *)(a1 + 88) )
        goto LABEL_4;
    }
    else if ( *(_QWORD *)(a1 + 88) )
    {
LABEL_4:
      v11 = *(_QWORD *)(a1 + 104);
      v87 = (*(_QWORD *)(a1 + 128) - *(_QWORD *)(a1 + 56)) / v11;
      v12 = (*(_QWORD *)(a1 + 136) - *(_QWORD *)(a1 + 56)) / v11 - v87;
      v13 = sub_127FA20(*(_QWORD *)a1, *(_QWORD *)(a1 + 96));
      v14 = (__int64 *)sub_1A22100(
                         a1,
                         *(__int64 ****)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
                         v13 >> 3,
                         a3,
                         a4,
                         a5);
      v15 = sub_1A1C950(*(_QWORD *)a1, v6, v14, *(_QWORD *)(a1 + 96));
      v16 = (__int64)v15;
      if ( v12 > 1 )
      {
        v44 = *(_QWORD **)(a1 + 216);
        v76 = (__int64 **)v15;
        v99 = 1;
        v97.m128i_i64[0] = (__int64)"vsplat";
        v98 = 3;
        v82 = (__int64 *)sub_1643350(v44);
        v45 = (__int64 **)sub_16463B0(*v76, v12);
        v46 = sub_1599EF0(v45);
        v103 = 1;
        v101.m128i_i64[0] = (__int64)".splatinsert";
        v73 = v46;
        v102 = 3;
        sub_14EC200(v100, &v97, &v101);
        v47 = sub_15A0680((__int64)v82, 0, 0);
        v77 = sub_1A1D560(v6, v73, (__int64)v76, v47, v100);
        v48 = (__int64 **)sub_16463B0(v82, v12);
        v49 = sub_1598F00(v48);
        v103 = 1;
        v50 = (_BYTE *)v49;
        v102 = 3;
        v101.m128i_i64[0] = (__int64)".splat";
        sub_14EC200(v100, &v97, &v101);
        v16 = sub_1A1D3A0(v6, v77, v73, v50, v100);
      }
      v17 = *(_QWORD *)(a1 + 48);
      v75 = v16;
      v18 = 1 << *(_WORD *)(v17 + 18);
      v80 = sub_1A1D0C0(v6, v17, "oldload");
      sub_15F8F50((__int64)v80, v18 >> 1);
      v103 = 1;
      v101.m128i_i64[0] = (__int64)"vec";
      v102 = 3;
      v19 = sub_1A1DB70((__int64)v6, v80, v75, v87, &v101, v75);
      goto LABEL_7;
    }
    if ( *(_QWORD *)(a1 + 80) )
      goto LABEL_25;
    if ( *(_QWORD *)(a1 + 112) <= *(_QWORD *)(a1 + 56) )
    {
      v90 = v10;
      if ( *(_QWORD *)(a1 + 120) >= *(_QWORD *)(a1 + 64) )
      {
        v83 = *(_QWORD *)(a1 + 144);
        if ( v83 == (unsigned __int64)(sub_127FA20(*(_QWORD *)a1, v9) + 7) >> 3 )
        {
          if ( (unsigned __int8)sub_1A1E0B0(v9) )
          {
            v55 = v90;
            v91 = *(_QWORD *)a1;
            v84 = v55;
            v56 = sub_127FA20(*(_QWORD *)a1, v55);
            v57 = *(unsigned __int8 **)(v91 + 24);
            v58 = &v57[*(unsigned int *)(v91 + 32)];
            if ( v57 != v58 )
            {
              while ( v56 != *v57 )
              {
                if ( v58 == ++v57 )
                  goto LABEL_46;
              }
              if ( (sub_127FA20(*(_QWORD *)a1, v84) & 7) == 0 )
              {
                if ( *(_QWORD *)(a1 + 88) )
                  goto LABEL_4;
                if ( !*(_QWORD *)(a1 + 80) )
                {
                  v59 = sub_127FA20(*(_QWORD *)a1, v84);
                  v60 = sub_1A22100(
                          a1,
                          *(__int64 ****)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
                          v59 >> 3,
                          a3,
                          a4,
                          a5);
                  v35 = (__int64 *)v60;
                  if ( *(_BYTE *)(v9 + 8) == 16 )
                  {
                    v78 = (__int64 **)v60;
                    v61 = *(_QWORD **)(a1 + 216);
                    v62 = *(_QWORD *)(v9 + 32);
                    v97.m128i_i64[0] = (__int64)"vsplat";
                    v99 = 1;
                    v85 = v62;
                    v98 = 3;
                    v92 = (__int64 *)sub_1643350(v61);
                    v63 = (__int64 **)sub_16463B0(*v78, v85);
                    v64 = sub_1599EF0(v63);
                    v103 = 1;
                    v101.m128i_i64[0] = (__int64)".splatinsert";
                    v74 = v64;
                    v102 = 3;
                    sub_14EC200(v100, &v97, &v101);
                    v65 = sub_15A0680((__int64)v92, 0, 0);
                    v79 = sub_1A1D560(v6, v74, (__int64)v78, v65, v100);
                    v66 = (__int64 **)sub_16463B0(v92, v85);
                    v67 = sub_1598F00(v66);
                    v103 = 1;
                    v86 = (_BYTE *)v67;
                    v101.m128i_i64[0] = (__int64)".splat";
                    v102 = 3;
                    sub_14EC200(v100, &v97, &v101);
                    v35 = (__int64 *)sub_1A1D3A0(v6, v79, v74, v86, v100);
                  }
                  goto LABEL_29;
                }
LABEL_25:
                v35 = (__int64 *)sub_1A22100(
                                   a1,
                                   *(__int64 ****)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
                                   *(_DWORD *)(a1 + 136) - *(_DWORD *)(a1 + 128),
                                   a3,
                                   a4,
                                   a5);
                if ( *(_QWORD *)(a1 + 80) )
                {
                  v36 = *(_QWORD *)(a1 + 56);
                  if ( *(_QWORD *)(a1 + 112) != v36 || v36 != *(_QWORD *)(a1 + 120) )
                  {
                    v37 = *(_QWORD *)(a1 + 48);
                    v81 = v35;
                    v88 = (unsigned int)(1 << *(_WORD *)(v37 + 18)) >> 1;
                    v38 = sub_1A1D0C0(v6, v37, "oldload");
                    v39 = v88;
                    v89 = v38;
                    sub_15F8F50((__int64)v38, v39);
                    v40 = sub_1A1C950(*(_QWORD *)a1, v6, v89, *(_QWORD *)(a1 + 80));
                    v41 = *(_BYTE **)a1;
                    v42 = *(_QWORD *)(a1 + 128);
                    v103 = 1;
                    v43 = v42 - *(_QWORD *)(a1 + 56);
                    v102 = 3;
                    v101.m128i_i64[0] = (__int64)"insert";
                    v35 = sub_1A202F0(v41, (__int64)v6, (__int64)v40, v81, v43, &v101, a3, a4, a5);
                  }
                }
LABEL_29:
                v19 = sub_1A1C950(*(_QWORD *)a1, v6, v35, v9);
LABEL_7:
                v20 = *(_QWORD *)(a2 + 24 * (3LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
                v21 = *(_DWORD *)(v20 + 32);
                if ( v21 <= 0x40 )
                  v22 = *(_QWORD *)(v20 + 24) == 0;
                else
                  v22 = v21 == (unsigned int)sub_16A57B0(v20 + 24);
                v23 = *(_QWORD *)(a1 + 48);
                v24 = 1 << *(_WORD *)(v23 + 18);
                v25 = sub_1A1CF60(v6, (__int64)v19, v23, !v22);
                sub_15F9450((__int64)v25, v24 >> 1);
                if ( v94 || v95 || v96 )
                  sub_1626170((__int64)v25, &v94);
                v26 = *(_QWORD *)(a2 + 24 * (3LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
                v27 = *(_DWORD *)(v26 + 32);
                if ( v27 <= 0x40 )
                  return *(_QWORD *)(v26 + 24) == 0;
                else
                  return v27 == (unsigned int)sub_16A57B0(v26 + 24);
              }
            }
          }
        }
      }
    }
LABEL_46:
    v68 = (__int64 *)sub_15A0680(
                       **(_QWORD **)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
                       *(_QWORD *)(a1 + 136) - *(_QWORD *)(a1 + 128),
                       0);
    v93 = sub_1A211D0(a2);
    v69 = sub_1A22080((__int64 *)a1, 0);
    v70 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    v71 = (_QWORD *)sub_1A246E0((__int64 *)a1, (__int64)v6, **(_QWORD **)(a1 + 168));
    v72 = sub_15E7280(v6, v71, v70, v68, v69, v93, 0, 0, 0);
    if ( v94 || v95 || (result = 0, v96) )
    {
      sub_1626170((__int64)v72, &v94);
      return 0;
    }
  }
  return result;
}
