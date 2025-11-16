// Function: sub_1167470
// Address: 0x1167470
//
unsigned __int8 *__fastcall sub_1167470(__m128i *a1, __int64 a2)
{
  __int64 v4; // rax
  char v5; // al
  __int64 v6; // rax
  __int64 v8; // r15
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  __int64 v11; // r8
  _BYTE *v12; // r11
  __int64 v13; // r10
  unsigned __int8 v14; // al
  __int64 v15; // r15
  unsigned int v16; // edi
  __int64 v17; // rsi
  __int64 v18; // rdx
  unsigned int **v19; // rdi
  _BYTE *v20; // rcx
  __int64 v21; // rdx
  unsigned int v22; // esi
  __int64 v23; // rax
  int v24; // eax
  unsigned int v25; // ebx
  __int64 v26; // rax
  unsigned int v27; // r8d
  __int64 v28; // rsi
  _QWORD *v29; // rax
  __int64 v30; // r10
  _BYTE *v31; // r11
  __int64 v32; // rax
  __int64 v33; // r8
  unsigned __int8 *v34; // r10
  __int64 v35; // r11
  __int64 v36; // rax
  __int64 v37; // rbx
  __int64 *v38; // r15
  __m128i v39; // rax
  __int64 v40; // rax
  __int64 v41; // rbx
  __int64 v42; // rdx
  _BYTE *v43; // rax
  unsigned int v44; // edi
  char v45; // al
  __int64 v46; // rsi
  char v47; // al
  unsigned __int8 v48; // al
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rdi
  bool v53; // al
  unsigned int **v54; // r12
  unsigned __int8 *v55; // rbx
  char v56; // al
  int v57; // eax
  int v58; // eax
  __int64 v59; // rax
  char v60; // al
  int v61; // eax
  int v62; // eax
  __int64 v63; // rsi
  __int64 v64; // rax
  __int64 v65; // rdx
  int v66; // eax
  __int64 v67; // rax
  __m128i *v68; // rdi
  __m128i *v69; // rsi
  __int64 i; // rcx
  __int64 v71; // rbx
  __int64 v72; // rax
  __int64 v73; // rax
  int v74; // [rsp+8h] [rbp-C8h]
  char v75; // [rsp+Fh] [rbp-C1h]
  _BYTE *v76; // [rsp+10h] [rbp-C0h]
  _BYTE *v77; // [rsp+10h] [rbp-C0h]
  _BYTE *v78; // [rsp+10h] [rbp-C0h]
  __int64 v79; // [rsp+10h] [rbp-C0h]
  _BYTE *v80; // [rsp+10h] [rbp-C0h]
  __int64 v81; // [rsp+10h] [rbp-C0h]
  _BYTE *v82; // [rsp+10h] [rbp-C0h]
  _BYTE *v83; // [rsp+10h] [rbp-C0h]
  __int64 v84; // [rsp+10h] [rbp-C0h]
  __int64 v85; // [rsp+18h] [rbp-B8h]
  __int64 v86; // [rsp+18h] [rbp-B8h]
  __int64 v87; // [rsp+18h] [rbp-B8h]
  bool v88; // [rsp+18h] [rbp-B8h]
  __int64 v89; // [rsp+18h] [rbp-B8h]
  _BYTE *v90; // [rsp+18h] [rbp-B8h]
  __int64 v91; // [rsp+18h] [rbp-B8h]
  __int64 v92; // [rsp+18h] [rbp-B8h]
  _BYTE *v93; // [rsp+18h] [rbp-B8h]
  __int64 v94; // [rsp+18h] [rbp-B8h]
  bool v95; // [rsp+27h] [rbp-A9h] BYREF
  __int64 v96; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v97; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v98; // [rsp+38h] [rbp-98h] BYREF
  __int64 *v99; // [rsp+40h] [rbp-90h] BYREF
  char v100; // [rsp+48h] [rbp-88h]
  __m128i v101; // [rsp+50h] [rbp-80h] BYREF
  __m128i v102; // [rsp+60h] [rbp-70h]
  unsigned __int64 v103; // [rsp+70h] [rbp-60h]
  __int64 v104; // [rsp+78h] [rbp-58h]
  __m128i v105; // [rsp+80h] [rbp-50h]
  __int64 v106; // [rsp+90h] [rbp-40h]

  v4 = a1[10].m128i_i64[0];
  v101 = _mm_loadu_si128(a1 + 6);
  v103 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v106 = v4;
  v104 = a2;
  v102 = _mm_loadu_si128(a1 + 7);
  v105 = _mm_loadu_si128(a1 + 9);
  v5 = sub_B44E60(a2);
  v6 = sub_101A990(*(__int64 ****)(a2 - 64), *(unsigned __int8 **)(a2 - 32), v5, &v101);
  if ( v6 )
    return sub_F162A0((__int64)a1, a2, v6);
  v8 = (__int64)sub_F0F270((__int64)a1, (unsigned __int8 *)a2);
  if ( !v8 )
  {
    v8 = (__int64)sub_1166190(a1, a2);
    if ( !v8 )
    {
      v8 = (__int64)sub_1160A10((__int64)a1, (unsigned __int8 *)a2, v9, v10, v11);
      if ( !v8 )
      {
        v12 = *(_BYTE **)(a2 - 64);
        v13 = *(_QWORD *)(a2 - 32);
        v102.m128i_i8[0] = 0;
        v101.m128i_i64[0] = (__int64)&v96;
        v101.m128i_i64[1] = (__int64)&v97;
        if ( *v12 == 55 )
        {
          if ( *((_QWORD *)v12 - 8) )
          {
            v96 = *((_QWORD *)v12 - 8);
            v81 = v13;
            v90 = v12;
            v45 = sub_991580((__int64)&v101.m128i_i64[1], *((_QWORD *)v12 - 4));
            v12 = v90;
            v13 = v81;
            if ( v45 )
            {
              v46 = v81;
              v82 = v90;
              v91 = v13;
              v99 = &v98;
              v100 = 0;
              v47 = sub_991580((__int64)&v99, v46);
              v13 = v91;
              v12 = v82;
              if ( v47 )
              {
                sub_C47CA0((__int64)&v99, v98, v97, &v95);
                if ( !v95 )
                {
                  if ( sub_B44E60(a2) )
                  {
                    v48 = *v82;
                    if ( *v82 <= 0x1Cu )
                    {
                      if ( v48 == 5 )
                      {
                        v66 = *((unsigned __int16 *)v82 + 1);
                        if ( (unsigned __int16)(v66 - 26) <= 1u || (unsigned int)(v66 - 19) <= 1 )
                        {
LABEL_46:
                          v49 = v96;
                          if ( (v82[1] & 2) != 0 )
                          {
                            LOWORD(v103) = 257;
                            v50 = sub_AD8D80(*(_QWORD *)(v96 + 8), (__int64)&v99);
                            v8 = sub_B504D0(19, v96, v50, (__int64)&v101, 0, 0);
                            sub_B448B0(v8, 1);
LABEL_48:
                            sub_969240((__int64 *)&v99);
                            return (unsigned __int8 *)v8;
                          }
LABEL_79:
                          LOWORD(v103) = 257;
                          v67 = sub_AD8D80(*(_QWORD *)(v49 + 8), (__int64)&v99);
                          v8 = sub_B504D0(19, v96, v67, (__int64)&v101, 0, 0);
                          goto LABEL_48;
                        }
                      }
                    }
                    else if ( (unsigned int)v48 - 48 <= 1 || (unsigned __int8)(v48 - 55) <= 1u )
                    {
                      goto LABEL_46;
                    }
                  }
                  v49 = v96;
                  goto LABEL_79;
                }
                sub_969240((__int64 *)&v99);
                v12 = v82;
                v13 = v91;
              }
            }
          }
        }
        v14 = *(_BYTE *)v13;
        v15 = *(_QWORD *)(a2 + 8);
        if ( *(_BYTE *)v13 == 17 )
        {
          v16 = *(_DWORD *)(v13 + 32);
          v17 = *(_QWORD *)(v13 + 24);
          v18 = 1LL << ((unsigned __int8)v16 - 1);
          if ( v16 > 0x40 )
            v17 = *(_QWORD *)(v17 + 8LL * ((v16 - 1) >> 6));
        }
        else
        {
          v41 = *(_QWORD *)(v13 + 8);
          v42 = (unsigned int)*(unsigned __int8 *)(v41 + 8) - 17;
          if ( (unsigned int)v42 > 1 || v14 > 0x15u )
          {
LABEL_26:
            if ( v14 != 69 )
              goto LABEL_27;
            v51 = *(_QWORD *)(v13 - 32);
            if ( !v51 )
              goto LABEL_27;
            v52 = *(_QWORD *)(v51 + 8);
            v96 = *(_QWORD *)(v13 - 32);
            if ( (unsigned int)*(unsigned __int8 *)(v52 + 8) - 17 <= 1 )
              v52 = **(_QWORD **)(v52 + 16);
            v83 = v12;
            v92 = v13;
            v53 = sub_BCAC40(v52, 1);
            v13 = v92;
            v12 = v83;
            if ( !v53 )
            {
LABEL_27:
              v77 = v12;
              v86 = v13;
              v29 = sub_11560B0((unsigned __int8 *)a2, (__int64)a1);
              v30 = v86;
              v31 = v77;
              v8 = (__int64)v29;
              if ( !v29 )
              {
                if ( *v77 != 55 )
                  goto LABEL_29;
                v55 = (unsigned __int8 *)*((_QWORD *)v77 - 8);
                if ( !v55 )
                  goto LABEL_29;
                v56 = sub_987880(*((unsigned __int8 **)v77 - 8));
                v31 = v77;
                if ( v56 )
                {
                  v57 = *v55;
                  v58 = (unsigned __int8)v57 <= 0x1Cu ? *((unsigned __int16 *)v55 + 1) : v57 - 29;
                  if ( v58 == 17 && (v55[1] & 2) != 0 )
                  {
                    v59 = *((_QWORD *)v55 - 8);
                    if ( v86 == v59 )
                    {
                      if ( v59 )
                      {
                        v63 = *((_QWORD *)v55 - 4);
                        if ( v63 )
                        {
                          v65 = *((_QWORD *)v77 - 4);
                          if ( v65 )
                            goto LABEL_71;
                        }
                      }
                    }
                  }
                }
                v60 = sub_987880(v55);
                v30 = v86;
                v31 = v77;
                if ( !v60 )
                  goto LABEL_29;
                v61 = *v55;
                v62 = (unsigned __int8)v61 <= 0x1Cu ? *((unsigned __int16 *)v55 + 1) : v61 - 29;
                if ( v62 == 17
                  && (v55[1] & 2) != 0
                  && (v63 = *((_QWORD *)v55 - 8)) != 0
                  && (v64 = *((_QWORD *)v55 - 4), v86 == v64)
                  && v64
                  && (v65 = *((_QWORD *)v77 - 4)) != 0 )
                {
LABEL_71:
                  LOWORD(v103) = 257;
                  v93 = v31;
                  v8 = sub_B504D0(26, v63, v65, (__int64)&v101, 0, 0);
                  if ( sub_B44E60(a2) && (v93[1] & 2) != 0 )
                    sub_B448B0(v8, 1);
                }
                else
                {
LABEL_29:
                  v78 = v31;
                  v87 = v30;
                  v32 = sub_1156740((__int64)a1, v30, 0, 1u, 0);
                  v34 = (unsigned __int8 *)v87;
                  v35 = (__int64)v78;
                  if ( v32 )
                  {
                    v36 = sub_1156740((__int64)a1, v87, 0, 1u, 1u);
                    v34 = (unsigned __int8 *)v87;
                    v35 = (__int64)v78;
                    v37 = v36;
                    if ( v36 )
                      goto LABEL_31;
                  }
                  v68 = &v101;
                  v69 = a1 + 6;
                  for ( i = 18; i; --i )
                  {
                    v68->m128i_i32[0] = v69->m128i_i32[0];
                    v69 = (__m128i *)((char *)v69 + 4);
                    v68 = (__m128i *)((char *)v68 + 4);
                  }
                  v84 = v35;
                  v104 = a2;
                  v94 = (__int64)v34;
                  if ( (unsigned __int8)sub_9A1DB0(v34, 1, 0, (__int64)&v101, v33) )
                  {
                    v71 = a1[2].m128i_i64[0];
                    LOWORD(v103) = 257;
                    HIDWORD(v99) = 0;
                    v72 = sub_ACD6D0(*(__int64 **)(v71 + 72));
                    v73 = sub_B33C40(v71, 0x43u, v94, v72, (__int64)v99, (__int64)&v101);
                    v35 = v84;
                    v37 = v73;
                    if ( v73 )
                    {
LABEL_31:
                      v79 = v35;
                      v38 = (__int64 *)a1[2].m128i_i64[0];
                      v88 = sub_B44E60(a2);
                      v39.m128i_i64[0] = (__int64)sub_BD5D20(a2);
                      LOWORD(v103) = 261;
                      v101 = v39;
                      v40 = sub_F94560(v38, v79, v37, (__int64)&v101, v88);
                      return sub_F162A0((__int64)a1, a2, v40);
                    }
                  }
                }
              }
              return (unsigned __int8 *)v8;
            }
            v54 = (unsigned int **)a1[2].m128i_i64[0];
            LOWORD(v103) = 257;
            v22 = 32;
            v19 = v54;
            v20 = (_BYTE *)sub_AD62B0(v15);
            v21 = (__int64)v83;
            goto LABEL_13;
          }
          v80 = v12;
          v89 = v13;
          v43 = sub_AD7630(v13, 0, v42);
          v13 = v89;
          v12 = v80;
          if ( !v43 || *v43 != 17 )
          {
            if ( *(_BYTE *)(v41 + 8) == 17 )
            {
              v24 = *(_DWORD *)(v41 + 32);
              v75 = 0;
              v25 = 0;
              v74 = v24;
              if ( v24 )
              {
                while ( 1 )
                {
                  v76 = v12;
                  v85 = v13;
                  v26 = sub_AD69F0((unsigned __int8 *)v13, v25);
                  v13 = v85;
                  v12 = v76;
                  if ( !v26 )
                    break;
                  if ( *(_BYTE *)v26 != 13 )
                  {
                    if ( *(_BYTE *)v26 != 17 )
                      break;
                    v27 = *(_DWORD *)(v26 + 32);
                    v28 = *(_QWORD *)(v26 + 24);
                    if ( v27 > 0x40 )
                      v28 = *(_QWORD *)(v28 + 8LL * ((v27 - 1) >> 6));
                    if ( (v28 & (1LL << ((unsigned __int8)v27 - 1))) == 0 )
                      break;
                    v75 = 1;
                  }
                  if ( v74 == ++v25 )
                  {
                    if ( v75 )
                      goto LABEL_12;
                    break;
                  }
                }
              }
            }
LABEL_25:
            v14 = *(_BYTE *)v13;
            goto LABEL_26;
          }
          v44 = *((_DWORD *)v43 + 8);
          v17 = *((_QWORD *)v43 + 3);
          v18 = 1LL << ((unsigned __int8)v44 - 1);
          if ( v44 > 0x40 )
            v17 = *(_QWORD *)(v17 + 8LL * ((v44 - 1) >> 6));
        }
        if ( (v17 & v18) != 0 )
        {
LABEL_12:
          v19 = (unsigned int **)a1[2].m128i_i64[0];
          v20 = (_BYTE *)v13;
          LOWORD(v103) = 257;
          v21 = (__int64)v12;
          v22 = 35;
LABEL_13:
          v23 = sub_92B530(v19, v22, v21, v20, (__int64)&v101);
          LOWORD(v103) = 257;
          return (unsigned __int8 *)sub_B520B0(v23, v15, (__int64)&v101, 0, 0);
        }
        goto LABEL_25;
      }
    }
  }
  return (unsigned __int8 *)v8;
}
