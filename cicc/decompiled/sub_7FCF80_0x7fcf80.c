// Function: sub_7FCF80
// Address: 0x7fcf80
//
_QWORD *__fastcall sub_7FCF80(__int64 a1, __int64 a2, __m128i *a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r13
  _QWORD *result; // rax
  __int64 v7; // r14
  int v8; // edx
  _QWORD *v9; // r15
  __int64 v10; // rax
  char v11; // al
  __m128i *v12; // r14
  __m128i *v13; // rax
  __int64 *v14; // rcx
  __m128i *v15; // rbx
  __m128i *v16; // rax
  __int64 j; // r14
  unsigned __int8 *v18; // r15
  __m128i *v19; // r13
  char v20; // r15
  char v21; // al
  __int64 *v22; // r13
  __int64 v23; // rax
  int v24; // edx
  _QWORD *v25; // r14
  __int64 *v26; // rdi
  _BYTE *v27; // rax
  __int64 v28; // r14
  unsigned int v29; // esi
  _QWORD *v30; // rbx
  _QWORD *v31; // rax
  __m128i *v32; // rax
  const __m128i *v33; // rsi
  __int64 *v34; // rcx
  _BYTE *v35; // rax
  void *v36; // r12
  _BYTE *v37; // rax
  _QWORD *v38; // r13
  _BYTE *v39; // rax
  _BYTE *v40; // r12
  __int64 k; // rax
  _BYTE *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 *v45; // r14
  const __m128i *v46; // r15
  _QWORD *v47; // rbx
  unsigned __int8 v48; // di
  __int64 v49; // r12
  __int64 v50; // rax
  _BYTE *v51; // r13
  __m128i v52; // xmm3
  __int64 v53; // rdi
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // rdi
  __int64 v57; // rdi
  __int64 v58; // r15
  __int64 v59; // r14
  __int64 v60; // rax
  __int64 v61; // r13
  __m128i *v62; // rax
  _BYTE *v63; // rax
  __int64 v64; // r12
  __int64 v65; // rdx
  _BYTE *v66; // rax
  void *v67; // rax
  __m128i *v68; // r13
  __int64 v69; // rax
  __m128i *v70; // r14
  _QWORD *v71; // rax
  const __m128i *v72; // rax
  __m128i *v73; // rax
  __int64 v74; // rax
  __m128i *v75; // rax
  bool v76; // zf
  __m128i *v77; // rax
  _BYTE *v78; // rax
  __int64 v79; // rax
  __int64 v80; // rsi
  __int64 v81; // [rsp+8h] [rbp-348h]
  __int64 v83; // [rsp+18h] [rbp-338h]
  __int64 v84; // [rsp+20h] [rbp-330h]
  __int64 v85; // [rsp+30h] [rbp-320h]
  int v86; // [rsp+3Ch] [rbp-314h]
  __int64 v87; // [rsp+40h] [rbp-310h]
  __m128i *v88; // [rsp+48h] [rbp-308h]
  __int64 v89; // [rsp+50h] [rbp-300h]
  int v90; // [rsp+58h] [rbp-2F8h]
  _QWORD *v91; // [rsp+58h] [rbp-2F8h]
  __int64 v92; // [rsp+68h] [rbp-2E8h]
  __int64 v93; // [rsp+70h] [rbp-2E0h]
  __m128i *v94; // [rsp+70h] [rbp-2E0h]
  __int64 v95; // [rsp+70h] [rbp-2E0h]
  _BYTE *v96; // [rsp+70h] [rbp-2E0h]
  __int64 v97; // [rsp+78h] [rbp-2D8h]
  __int64 v98; // [rsp+80h] [rbp-2D0h]
  __m128i *v99; // [rsp+88h] [rbp-2C8h]
  char v100; // [rsp+93h] [rbp-2BDh]
  int v101; // [rsp+94h] [rbp-2BCh]
  __int64 v103; // [rsp+A0h] [rbp-2B0h]
  __int64 v104; // [rsp+A0h] [rbp-2B0h]
  __int64 v105; // [rsp+A0h] [rbp-2B0h]
  __int64 *v106; // [rsp+A8h] [rbp-2A8h]
  unsigned int v107; // [rsp+B0h] [rbp-2A0h] BYREF
  char v108[4]; // [rsp+B4h] [rbp-29Ch] BYREF
  char v109[4]; // [rsp+B8h] [rbp-298h] BYREF
  int v110; // [rsp+BCh] [rbp-294h] BYREF
  __m128i *v111; // [rsp+C0h] [rbp-290h] BYREF
  __int64 *i; // [rsp+C8h] [rbp-288h] BYREF
  __m128i v113[2]; // [rsp+D0h] [rbp-280h] BYREF
  int v114; // [rsp+F0h] [rbp-260h] BYREF
  __int64 *v115; // [rsp+F8h] [rbp-258h]
  __m128i v116[2]; // [rsp+110h] [rbp-240h] BYREF
  char v117[96]; // [rsp+130h] [rbp-220h] BYREF
  _BYTE v118[192]; // [rsp+190h] [rbp-1C0h] BYREF
  __m128i v119[9]; // [rsp+250h] [rbp-100h] BYREF
  __int64 v120; // [rsp+2E8h] [rbp-68h]
  __int64 v121; // [rsp+2F0h] [rbp-60h]

  v3 = a1;
  v4 = a2;
  v5 = *(_QWORD *)(a1 + 152);
  v111 = 0;
  for ( i = 0; *(_BYTE *)(v5 + 140) == 12; v5 = *(_QWORD *)(v5 + 160) )
    ;
  result = (_QWORD *)*(unsigned int *)(a2 + 160);
  v101 = (int)result;
  if ( !(_DWORD)result )
  {
    v7 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
    v8 = sub_860080(a1, 0);
    if ( v8 )
      v8 = *(_DWORD *)(a1 + 164);
    v100 = *(_BYTE *)(a2 + 174);
    v97 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + 168LL);
    v89 = *(_QWORD *)(*(_QWORD *)(a2 + 152) + 168LL);
    v9 = *(_QWORD **)v89;
    if ( v100 == 6 )
    {
      v34 = *(__int64 **)v97;
      if ( (*(_BYTE *)(v97 + 16) & 0xC0) == 0x40 )
        v92 = *(_QWORD *)(*v34 + 8);
      else
        v92 = v34[1];
      v106 = sub_7F54F0(a2, 0, v8, &v107);
      sub_7E1740(v106[10], (__int64)v113);
      sub_7F6C60((__int64)v106, v107, (__int64)v118);
      v99 = 0;
    }
    else
    {
      v10 = v9[1];
      v9 = (_QWORD *)*v9;
      v92 = v10;
      v106 = sub_7F54F0(a2, 0, v8, &v107);
      sub_7E1740(v106[10], (__int64)v113);
      sub_7F6C60((__int64)v106, v107, (__int64)v118);
      v99 = sub_7E2270(v92);
      v106[5] = (__int64)v99;
      v106[8] = (__int64)v99;
      v99[10].m128i_i8[12] |= 1u;
    }
    if ( (*(_BYTE *)(a1 + 194) & 0x20) == 0
      && (*(_BYTE *)(v7 + 176) & 0x10) != 0
      && (*(_BYTE *)(a2 + 205) & 0x1C) == 4
      && (unsigned __int8)(*(_BYTE *)(a2 + 174) - 1) <= 1u )
    {
      v72 = (const __m128i *)sub_7E1DC0();
      v73 = sub_73C570(v72, 1);
      v74 = sub_72D2E0(v73);
      v88 = sub_7E7CA0(v74);
      v75 = (__m128i *)sub_73E830((__int64)v88);
      v76 = *(_BYTE *)(a2 + 174) == 2;
      v90 = 1;
      v111 = v75;
      i = (__int64 *)v75;
      v86 = 1;
      if ( v76 )
      {
        v77 = sub_7F7020(v7);
        v78 = sub_7E2510((__int64)v77, 0);
        *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v7 + 168) + 192LL) + 88LL) |= 4u;
        sub_7E6AB0((__int64)v88, (__int64)v78, v113[0].m128i_i32);
        v86 = 0;
        v101 = 1;
      }
LABEL_15:
      v12 = v99;
      if ( v9 )
      {
        do
        {
          while ( 1 )
          {
            v15 = v12;
            v16 = sub_7E2270(v9[1]);
            v16[8].m128i_i64[0] = (__int64)v9;
            v12 = v16;
            if ( v15 )
              v15[7].m128i_i64[0] = (__int64)v16;
            else
              v106[5] = (__int64)v16;
            v13 = (__m128i *)sub_73E830((__int64)v16);
            if ( !v111 )
              break;
            v14 = i;
            i = (__int64 *)v13;
            v14[2] = (__int64)v13;
            v9 = (_QWORD *)*v9;
            if ( !v9 )
              goto LABEL_27;
          }
          v111 = v13;
          i = (__int64 *)v13;
          v9 = (_QWORD *)*v9;
        }
        while ( v9 );
LABEL_27:
        v4 = a2;
      }
      if ( a3 )
      {
        for ( j = **(_QWORD **)v97; j; j = *(_QWORD *)j )
        {
          if ( (*(_BYTE *)(j + 32) & 4) != 0 )
            break;
        }
        v93 = qword_4F06BC0;
        sub_733780(0, 0, 0, 4, 0);
        v18 = (unsigned __int8 *)qword_4F06BC0;
        qword_4F06BC0 = v93;
        v98 = (__int64)v18;
        sub_7E18E0((__int64)v117, 0, (__int64)v18);
        a3 = (__m128i *)sub_73F6B0(a3, (a3[-1].m128i_i8[8] & 1) == 0 ? 520 : 8);
        if ( (unsigned int)sub_733920((__int64)v18) )
        {
          sub_7E1AA0();
          v98 = 0;
        }
        else
        {
          if ( dword_4D03F8C )
          {
            v96 = sub_726B30(11);
            sub_7E6810((__int64)v96, (__int64)v113, 1);
            sub_7E1740((__int64)v96, (__int64)v113);
            sub_732E60(v18, 0x14u, *((_QWORD **)v96 + 10));
          }
          sub_7E9190((__int64)v18, (__int64)v113);
        }
        sub_7E2BA0((__int64)v113);
        sub_7F1A60(a3, v5, a1, j, 0, 0, 0, 0);
        sub_7FAFA0((__int64)v113);
      }
      else
      {
        v98 = 0;
      }
      v94 = v111;
      if ( v111 )
        i[2] = (__int64)a3;
      else
        v94 = a3;
      if ( v90 )
      {
        v43 = sub_7FDF40(a1, 1, 0);
        v91 = (_QWORD *)sub_72B840(v43);
        v44 = sub_72B840(a1);
        v85 = v44;
        v45 = (__int64 *)(v44 + 48);
        v104 = *(_QWORD *)(v44 + 32);
        if ( *(_QWORD *)(v44 + 48) )
        {
          v81 = v4;
          v46 = *(const __m128i **)(v44 + 48);
          v47 = 0;
          do
          {
            v48 = v46->m128i_u8[8];
            if ( v48 )
            {
              v45 = (__int64 *)v46;
            }
            else
            {
              v49 = 0;
              if ( *(_BYTE *)(v104 + 174) == 2 )
              {
                v50 = v91[11];
                v49 = *(_QWORD *)(v50 + 24);
                *(_QWORD *)(v50 + 24) = 0;
                v48 = v46->m128i_u8[8];
              }
              v51 = sub_726BB0(v48);
              *(__m128i *)v51 = _mm_loadu_si128(v46);
              *((__m128i *)v51 + 1) = _mm_loadu_si128(v46 + 1);
              *((__m128i *)v51 + 2) = _mm_loadu_si128(v46 + 2);
              v52 = _mm_loadu_si128(v46 + 3);
              *(_QWORD *)v51 = 0;
              *((__m128i *)v51 + 3) = v52;
              v53 = v46[1].m128i_i64[1];
              if ( v53 )
                *((_QWORD *)v51 + 3) = sub_740B80(v53, 0x200u);
              *(__m128i *)(v51 + 40) = _mm_loadu_si128((const __m128i *)&unk_4F07370);
              if ( v46->m128i_i8[8] > 3u )
                sub_721090();
              v87 = *((_QWORD *)v51 + 3);
              if ( v87 )
              {
                v83 = v91[5];
                v84 = *(_QWORD *)(v85 + 40);
                sub_76C7C0((__int64)v119);
                v119[0].m128i_i64[0] = (__int64)sub_7F4FE0;
                v120 = v84;
                v121 = v83;
                sub_76D400(v87, (__int64)v119, v54, v55, v87);
              }
              if ( *(_BYTE *)(v104 + 174) == 2 && v49 )
              {
                v79 = v49;
                do
                {
                  v80 = v79;
                  v79 = *(_QWORD *)(v79 + 32);
                }
                while ( v79 );
                *(_QWORD *)(v80 + 32) = *(_QWORD *)(v91[11] + 24LL);
                *(_QWORD *)(v91[11] + 24LL) = v49;
              }
              if ( v47 )
                *v47 = v51;
              else
                v91[6] = v51;
              v47 = v51;
              *v45 = v46->m128i_i64[0];
              sub_733B20((_QWORD *)v46[1].m128i_i64[1]);
            }
            v46 = (const __m128i *)v46->m128i_i64[0];
          }
          while ( v46 );
          v3 = a1;
          v4 = v81;
        }
        v56 = *(_QWORD *)(v85 + 88);
        if ( v56 )
        {
          if ( (unsigned int)sub_733920(v56) )
          {
            v57 = *(_QWORD *)(v85 + 88);
            if ( (*(_BYTE *)(v57 + 1) & 4) == 0 )
            {
              sub_733650(v57);
              *(_QWORD *)(v85 + 88) = 0;
            }
          }
        }
        sub_7E91D0(v106[11], (__int64)v113);
        if ( v86 )
        {
          v58 = v106[5];
          v59 = v106[6];
          v60 = *(_QWORD *)(v106[4] + 40);
          v61 = *(_QWORD *)(v60 + 32);
          if ( v88 )
          {
            v62 = sub_7F7020(*(_QWORD *)(v60 + 32));
            v63 = sub_7E2510((__int64)v62, (__int64)v113);
            *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v61 + 168) + 192LL) + 88LL) |= 4u;
            sub_7E6AB0((__int64)v88, (__int64)v63, v113[0].m128i_i32);
          }
          sub_7F5B50(v61, v58, (__int64)v88, 0, v113[0].m128i_i32);
          if ( v59 )
          {
            v105 = v3;
            v64 = v59;
            do
            {
              if ( *(_BYTE *)(v64 + 8) )
                break;
              sub_801720(v64, v58, 1, v88, v113);
              v65 = *(_QWORD *)(v64 + 16);
              if ( *(_QWORD *)(*(_QWORD *)(v61 + 168) + 80LL) == v65 && v65 )
                sub_7F5B50(v61, v58, (__int64)v88, 0, v113[0].m128i_i32);
              v64 = *(_QWORD *)v64;
            }
            while ( v64 );
            v3 = v105;
          }
        }
      }
      if ( v100 == 6 )
      {
        v33 = (const __m128i *)sub_724DC0();
        v119[0].m128i_i64[0] = (__int64)v33;
        sub_72BB40(v92, v33);
        v19 = (__m128i *)sub_73A720((const __m128i *)v119[0].m128i_i64[0], (__int64)v33);
        sub_724E30((__int64)v119);
      }
      else
      {
        v19 = (__m128i *)sub_73E830((__int64)v99);
      }
      if ( (*(_BYTE *)(v97 + 16) & 0xC0) == 0x40 )
      {
        v19[1].m128i_i64[0] = v94[1].m128i_i64[0];
        v94[1].m128i_i64[0] = (__int64)v19;
        v19 = v94;
      }
      else
      {
        v19[1].m128i_i64[0] = (__int64)v94;
      }
      v20 = 0;
      v103 = 0;
      v21 = *(_BYTE *)(v4 + 205) & 0x1C;
      if ( *(_BYTE *)(v4 + 174) == 2 && v21 == 12 )
      {
        v103 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v4 + 40) + 32LL) + 168LL) + 184LL);
        if ( (unsigned int)sub_87ADD0(v103, v108, v109, &v110) && v110 )
        {
          v20 = 1;
          v22 = 0;
          goto LABEL_48;
        }
        v20 = 1;
        v21 = *(_BYTE *)(v4 + 205) & 0x1C;
      }
      if ( v21 == 16 )
      {
        sub_7E1790((__int64)&v114);
        v66 = sub_73E830(*(_QWORD *)(v106[5] + 112));
        v67 = sub_7F0830(v66);
        sub_7F8BA0((__int64)v67, 0, &v114, 0, (__int64)v116, (__int64)v119);
        sub_7F88F0(v3, v19, 0, v116);
        v68 = (__m128i *)sub_73E830((__int64)v99);
        v69 = sub_7FDF40(v3, 1, 0);
        sub_7F88F0(v69, v68, 0, v119);
        v22 = v115;
      }
      else
      {
        v22 = sub_7F88E0(v3, v19);
      }
LABEL_48:
      v23 = sub_7F8700(*(_QWORD *)(v4 + 152));
      v24 = sub_8D2600(v23);
      if ( v101 && (v25 = (_QWORD *)v106[6]) != 0 )
      {
        if ( !v22 )
          goto LABEL_66;
      }
      else
      {
        if ( (*(_BYTE *)(v89 + 16) & 1) == 0 && !v98 )
        {
          if ( !v24 )
          {
LABEL_57:
            v27 = sub_726B30(8);
            *((_QWORD *)v27 + 6) = v22;
            v28 = (__int64)v27;
            sub_7E6810((__int64)v27, (__int64)v113, 1);
            sub_7E17A0(v28);
            if ( *(_BYTE *)(v4 + 174) == 2 && (*(_BYTE *)(v4 + 205) & 0x1C) == 0xC && (*(_BYTE *)(v3 + 192) & 2) == 0 )
            {
              v35 = sub_73E830(v106[5]);
              v36 = sub_7F0830(v35);
              v37 = sub_726B30(1);
              *((_QWORD *)v37 + 6) = v36;
              v38 = v37;
              v39 = sub_726B30(11);
              v38[9] = v39;
              v40 = v39;
              *((_QWORD *)v39 + 9) = *(_QWORD *)(v106[10] + 72);
              *(_BYTE *)(*((_QWORD *)v39 + 10) + 24LL) &= ~1u;
              *(_QWORD *)(v106[10] + 72) = v38;
              v38[3] = v106[10];
              sub_7F64C0((__int64)v39, (__int64)v38);
              for ( k = *((_QWORD *)v40 + 9); k; k = *(_QWORD *)(k + 16) )
                *(_QWORD *)(k + 24) = v40;
              if ( !v38[2] )
              {
                v42 = sub_726B30(8);
                v38[2] = v42;
                sub_7E17A0((__int64)v42);
              }
            }
            v29 = v107;
            v106[6] = 0;
            result = sub_7FB010((__int64)v106, v29, (__int64)v118);
            if ( *(char *)(v4 + 192) < 0 )
            {
              result = (_QWORD *)dword_4D04380;
              if ( dword_4D04380 )
                return sub_76FD50(v106);
            }
            return result;
          }
          if ( !v22 )
          {
LABEL_53:
            if ( !v101 || (v25 = (_QWORD *)v106[6]) == 0 )
            {
              if ( !v20 )
                goto LABEL_55;
              goto LABEL_69;
            }
LABEL_66:
            v95 = v4;
            v30 = v25;
            do
            {
              sub_7FCE40((__int64)v30, (__int64)v99, 0, 1, (__int64)v88, (__int64)v113);
              v30 = (_QWORD *)*v30;
            }
            while ( v30 );
            v4 = v95;
            if ( !v20 )
            {
LABEL_55:
              if ( v98 )
              {
                sub_7E7530(v98, (__int64)v113);
                sub_7E1AA0();
              }
              goto LABEL_57;
            }
LABEL_69:
            v31 = sub_73E830((__int64)v99);
            v32 = (__m128i *)sub_7F6190(v103, *(_QWORD *)(*(_QWORD *)(v4 + 40) + 32LL), v31);
            sub_7F88F0(v103, v32, 0, v113);
            goto LABEL_55;
          }
LABEL_52:
          v26 = v22;
          v22 = 0;
          sub_7E69E0(v26, v113[0].m128i_i32);
          goto LABEL_53;
        }
        if ( !v22 )
          goto LABEL_53;
      }
      if ( !v24 )
      {
        v70 = sub_7E7CA0(*v22);
        v71 = (_QWORD *)sub_7E2BE0((__int64)v70, (__int64)v22);
        sub_7E69E0(v71, v113[0].m128i_i32);
        v22 = sub_73E830((__int64)v70);
        goto LABEL_53;
      }
      goto LABEL_52;
    }
    v11 = *(_BYTE *)(a1 + 174);
    if ( v11 == 1 )
    {
      if ( *(_BYTE *)(a2 + 174) != 1
        || (((*(_BYTE *)(a2 + 205) & 0x1C) - 8) & 0xF4) != 0
        || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 32LL) + 176LL) & 0x10) == 0 )
      {
        sub_7F9BA0(a1, 0, &v111, &i);
        v90 = 0;
        v86 = 0;
        v88 = 0;
        goto LABEL_15;
      }
    }
    else if ( v11 == 2
           && (*(_BYTE *)(a2 + 174) != 2
            || (((*(_BYTE *)(a2 + 205) & 0x1C) - 8) & 0xF4) != 0
            || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 32LL) + 176LL) & 0x10) == 0) )
    {
      sub_7F9C50(a1, 1, &v111, &i);
      v90 = 0;
      v86 = 0;
      v88 = 0;
      goto LABEL_15;
    }
    v90 = 0;
    v86 = 0;
    v88 = 0;
    goto LABEL_15;
  }
  return result;
}
