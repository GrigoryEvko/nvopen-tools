// Function: sub_2B7F570
// Address: 0x2b7f570
//
_BYTE *__fastcall sub_2B7F570(__int64 **a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // r12
  __int64 *v5; // rdi
  char v6; // cl
  __int64 *v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // r13
  __int64 *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  unsigned int v13; // edx
  __int64 *v14; // rsi
  __int64 v15; // r8
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r8
  unsigned int v19; // edx
  __int64 *v20; // r14
  __int64 v21; // rsi
  _QWORD *v22; // r12
  unsigned __int8 v23; // al
  __int64 *v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rdx
  _BYTE *v27; // r13
  int v29; // esi
  __int64 *v30; // r14
  __int64 v31; // rax
  __int64 *v32; // rcx
  __int64 *v33; // r13
  unsigned __int64 v34; // rax
  char v35; // al
  __int64 v36; // rax
  __int64 v37; // r14
  _QWORD *v38; // r15
  unsigned int v39; // esi
  __int64 v40; // r9
  unsigned int v41; // eax
  __int64 v42; // rdx
  __int64 v43; // r8
  __int64 v44; // rdi
  unsigned int v45; // esi
  __int64 v46; // r14
  __int64 v47; // rax
  unsigned int v48; // edx
  __int64 *v49; // r8
  __int64 v50; // rcx
  __int64 v51; // rbx
  unsigned int v53; // esi
  __int64 v54; // rdi
  unsigned int v55; // ecx
  _BYTE **v56; // rdx
  int v57; // r10d
  __int64 v58; // r9
  int v59; // eax
  int v60; // edx
  _QWORD *v61; // rax
  _BYTE *v62; // rdx
  __int64 v63; // rsi
  __int64 *v64; // r12
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // rsi
  int v68; // edx
  __int64 *v69; // rax
  _QWORD *v70; // rax
  __int64 v71; // r13
  __int64 v72; // r14
  __int64 v73; // rdx
  unsigned int v74; // esi
  int v75; // r10d
  int v76; // r10d
  __int64 v77; // r15
  __int64 v78; // rax
  __int64 v79; // rdx
  __int64 v80; // r13
  _QWORD *v81; // rax
  __int64 v82; // r13
  __int64 v83; // r14
  __int64 v84; // rdx
  unsigned int v85; // esi
  __int64 v86; // rax
  __int64 v87; // r15
  int v88; // r11d
  __int64 v89; // r10
  int v90; // eax
  __int64 *v91; // rax
  __int64 v92; // rdx
  __int64 v93; // r11
  int v94; // eax
  int v95; // edx
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 *v98; // rax
  __int64 v99; // [rsp+10h] [rbp-D0h]
  int v100; // [rsp+10h] [rbp-D0h]
  __int64 *v101; // [rsp+10h] [rbp-D0h]
  _BYTE *v102; // [rsp+18h] [rbp-C8h] BYREF
  _QWORD *v103; // [rsp+28h] [rbp-B8h] BYREF
  __int64 v104[4]; // [rsp+30h] [rbp-B0h] BYREF
  __int16 v105; // [rsp+50h] [rbp-90h]
  __m128i v106; // [rsp+60h] [rbp-80h] BYREF
  __int64 v107; // [rsp+70h] [rbp-70h]
  __int64 v108; // [rsp+78h] [rbp-68h]
  __int64 v109; // [rsp+80h] [rbp-60h]
  __int64 v110; // [rsp+88h] [rbp-58h]
  __int64 v111; // [rsp+90h] [rbp-50h]
  __int64 v112; // [rsp+98h] [rbp-48h]
  __int16 v113; // [rsp+A0h] [rbp-40h]

  v3 = *a1;
  v102 = (_BYTE *)a2;
  v4 = *v3;
  if ( *(_QWORD *)(a2 + 8) != *(_QWORD *)(*v3 + 8) )
  {
    if ( *(_BYTE *)v4 <= 0x1Cu )
    {
      v9 = 0;
      v6 = 0;
    }
    else
    {
      v5 = a1[1];
      v6 = *((_BYTE *)v5 + 2788);
      if ( v6 )
      {
        v7 = (__int64 *)v5[346];
        v8 = &v7[*((unsigned int *)v5 + 695)];
        if ( v7 == v8 )
        {
LABEL_67:
          v9 = v4;
          v6 = 0;
        }
        else
        {
          while ( 1 )
          {
            v9 = *v7;
            if ( v4 == *v7 )
              break;
            if ( v8 == ++v7 )
              goto LABEL_67;
          }
        }
      }
      else
      {
        v9 = *v3;
        v6 = sub_C8CA60((__int64)(v5 + 345), *v3) != 0;
        v4 = **a1;
      }
    }
    v10 = a1[2];
    v11 = v10[1];
    v12 = *((unsigned int *)v10 + 6);
    if ( (_DWORD)v12 )
    {
      v13 = (v12 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v14 = (__int64 *)(v11 + 40LL * v13);
      v15 = *v14;
      if ( v4 == *v14 )
      {
LABEL_10:
        if ( v14 != (__int64 *)(v11 + 40 * v12) )
        {
          v16 = v6 ? *(_QWORD *)(v9 + 40) : a1[1][427];
          v17 = *((unsigned int *)v14 + 8);
          v18 = v14[2];
          if ( (_DWORD)v17 )
          {
            v19 = (v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
            v20 = (__int64 *)(v18 + 24LL * v19);
            v21 = *v20;
            if ( v16 == *v20 )
            {
LABEL_15:
              if ( v20 != (__int64 *)(v18 + 24 * v17) )
              {
                v22 = (_QWORD *)v20[1];
                v23 = *(_BYTE *)v22;
                if ( *(_BYTE *)v22 <= 0x1Cu || v6 || (v24 = a1[1], v25 = v24[428], v25 == v24[427] + 48) )
                {
LABEL_26:
                  v27 = (_BYTE *)v20[2];
                  if ( !v27 )
                    v27 = v22;
                  goto LABEL_28;
                }
                if ( v25 )
                  v25 -= 24;
                if ( !sub_B445A0(v25, v20[1]) )
                {
LABEL_25:
                  v23 = *(_BYTE *)v22;
                  goto LABEL_26;
                }
                v26 = a1[1][428];
                if ( !v26 )
                  BUG();
                sub_B44550(v22, *(_QWORD *)(v26 + 16), (unsigned __int64 *)v26, *((unsigned __int16 *)a1[1] + 1716));
                v27 = (_BYTE *)v20[2];
                if ( *v27 > 0x1Cu )
                {
                  sub_B44530((_QWORD *)v20[2], (__int64)v22);
                  goto LABEL_25;
                }
LABEL_48:
                v23 = *(_BYTE *)v22;
LABEL_28:
                if ( v23 > 0x1Cu )
                {
                  v103 = v22;
                  if ( *(_BYTE *)v22 != 84 && !(unsigned __int8)sub_991AB0((char *)v22) )
                  {
                    v64 = a1[1];
                    sub_2400480((__int64)&v106, (__int64)(v64 + 389), (__int64 *)&v103);
                    if ( (_BYTE)v109 )
                    {
                      v86 = *((unsigned int *)v64 + 788);
                      v87 = (__int64)v103;
                      if ( v86 + 1 > (unsigned __int64)*((unsigned int *)v64 + 789) )
                      {
                        sub_C8D5F0((__int64)(v64 + 393), v64 + 395, v86 + 1, 8u, v65, v66);
                        v86 = *((unsigned int *)v64 + 788);
                      }
                      *(_QWORD *)(v64[393] + 8 * v86) = v87;
                      ++*((_DWORD *)v64 + 788);
                    }
                    v67 = (__int64)(a1[1] + 395);
                    v104[0] = v103[5];
                    sub_29B09C0((__int64)&v106, v67, v104);
                  }
                }
                return v27;
              }
            }
            else
            {
              v76 = 1;
              while ( v21 != -4096 )
              {
                v19 = (v17 - 1) & (v76 + v19);
                v20 = (__int64 *)(v18 + 24LL * v19);
                v21 = *v20;
                if ( v16 == *v20 )
                  goto LABEL_15;
                ++v76;
              }
            }
          }
        }
      }
      else
      {
        v29 = 1;
        while ( v15 != -4096 )
        {
          v75 = v29 + 1;
          v13 = (v12 - 1) & (v29 + v13);
          v14 = (__int64 *)(v11 + 40LL * v13);
          v15 = *v14;
          if ( *v14 == v4 )
            goto LABEL_10;
          v29 = v75;
        }
      }
    }
    if ( v6 )
    {
      if ( *(_BYTE *)v9 == 90 )
      {
        v63 = (__int64)a1[3];
        v104[0] = v9;
        sub_2B7F1E0((__int64)&v106, v63, v104);
        v22 = (_QWORD *)v104[0];
      }
      else
      {
        v104[0] = 0;
        v22 = (_QWORD *)sub_B47F80((_BYTE *)v9);
        sub_B44220(v22, v9 + 24, 0);
        if ( (*(_BYTE *)(v9 + 7) & 0x10) != 0 )
          sub_BD6B90((unsigned __int8 *)v22, (unsigned __int8 *)v9);
      }
    }
    else
    {
      v30 = a1[1];
      if ( *(_BYTE *)v4 == 90 && *v102 > 0x1Cu )
      {
        v77 = *(_QWORD *)(v4 - 64);
        v78 = sub_2B2A0E0((__int64)a1[1], v77);
        if ( v79 )
          v77 = *(_QWORD *)(*(_QWORD *)v78 + 96LL);
        if ( *(_BYTE *)v77 > 0x1Cu && (_BYTE *)v77 != v102 && *((_QWORD *)v102 + 5) == *(_QWORD *)(v77 + 40) )
        {
          if ( !sub_B445A0(v77, (__int64)v102) )
          {
            v98 = a1[1];
            LOWORD(v109) = 257;
            v22 = (_QWORD *)sub_A837F0((unsigned int **)v98 + 421, v102, (_BYTE *)*a1[4], (__int64)&v106);
            goto LABEL_38;
          }
          v30 = a1[1];
        }
        v105 = 257;
        v80 = *(_QWORD *)(v4 - 32);
        v22 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v30[431] + 96LL))(
                          v30[431],
                          v77,
                          v80);
        if ( !v22 )
        {
          LOWORD(v109) = 257;
          v81 = sub_BD2C40(72, 2u);
          v22 = v81;
          if ( v81 )
            sub_B4DE80((__int64)v81, v77, v80, (__int64)&v106, 0, 0);
          (*(void (__fastcall **)(__int64, _QWORD *, __int64 *, __int64, __int64))(*(_QWORD *)v30[432] + 16LL))(
            v30[432],
            v22,
            v104,
            v30[428],
            v30[429]);
          v82 = v30[421];
          v83 = v82 + 16LL * *((unsigned int *)v30 + 844);
          while ( v83 != v82 )
          {
            v84 = *(_QWORD *)(v82 + 8);
            v85 = *(_DWORD *)v82;
            v82 += 16;
            sub_B99FD0((__int64)v22, v85, v84);
          }
        }
      }
      else
      {
        v31 = *(_QWORD *)(v4 + 8);
        if ( *(_BYTE *)(v31 + 8) == 17 )
        {
          v22 = (_QWORD *)sub_2B1F140(
                            (__int64)(v30 + 421),
                            (__int64)v102,
                            *(_DWORD *)(v31 + 32),
                            *(_DWORD *)(v31 + 32) * *((_DWORD *)a1[5] + 6));
        }
        else
        {
          v69 = a1[4];
          v105 = 257;
          v99 = *v69;
          v22 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, _BYTE *))(*(_QWORD *)v30[431] + 96LL))(v30[431], v102);
          if ( !v22 )
          {
            LOWORD(v109) = 257;
            v70 = sub_BD2C40(72, 2u);
            v22 = v70;
            if ( v70 )
              sub_B4DE80((__int64)v70, (__int64)v102, v99, (__int64)&v106, 0, 0);
            (*(void (__fastcall **)(__int64, _QWORD *, __int64 *, __int64, __int64))(*(_QWORD *)v30[432] + 16LL))(
              v30[432],
              v22,
              v104,
              v30[428],
              v30[429]);
            v71 = v30[421];
            v72 = v71 + 16LL * *((unsigned int *)v30 + 844);
            while ( v72 != v71 )
            {
              v73 = *(_QWORD *)(v71 + 8);
              v74 = *(_DWORD *)v71;
              v71 += 16;
              sub_B99FD0((__int64)v22, v74, v73);
            }
          }
        }
      }
    }
LABEL_38:
    v32 = *a1;
    v27 = v22;
    if ( v22[1] != *(_QWORD *)(**a1 + 8) )
    {
      v33 = a1[1];
      v105 = 257;
      v34 = v33[418];
      v113 = 257;
      v106 = (__m128i)v34;
      v107 = 0;
      v108 = 0;
      v109 = 0;
      v110 = 0;
      v111 = 0;
      v112 = 0;
      v35 = sub_9AC470(*v32, &v106, 0);
      v36 = sub_921630((unsigned int **)v33 + 421, (__int64)v22, *(_QWORD *)(**a1 + 8), v35 ^ 1u, (__int64)v104);
      v32 = *a1;
      v27 = (_BYTE *)v36;
    }
    v37 = (__int64)a1[2];
    v38 = 0;
    if ( *(_BYTE *)v22 >= 0x1Du )
      v38 = v22;
    v39 = *(_DWORD *)(v37 + 24);
    if ( v39 )
    {
      v40 = *(_QWORD *)(v37 + 8);
      v41 = (v39 - 1) & (((unsigned int)*v32 >> 9) ^ ((unsigned int)*v32 >> 4));
      v42 = v40 + 40LL * v41;
      v43 = *(_QWORD *)v42;
      if ( *(_QWORD *)v42 == *v32 )
      {
LABEL_44:
        v44 = *(_QWORD *)(v42 + 16);
        v45 = *(_DWORD *)(v42 + 32);
        v46 = v42 + 8;
        goto LABEL_45;
      }
      v100 = 1;
      v93 = 0;
      while ( v43 != -4096 )
      {
        if ( !v93 && v43 == -8192 )
          v93 = v42;
        v41 = (v39 - 1) & (v100 + v41);
        v42 = v40 + 40LL * v41;
        v43 = *(_QWORD *)v42;
        if ( *v32 == *(_QWORD *)v42 )
          goto LABEL_44;
        ++v100;
      }
      if ( !v93 )
        v93 = v42;
      v106.m128i_i64[0] = v93;
      v94 = *(_DWORD *)(v37 + 16);
      ++*(_QWORD *)v37;
      v95 = v94 + 1;
      if ( 4 * (v94 + 1) < 3 * v39 )
      {
        if ( v39 - *(_DWORD *)(v37 + 20) - v95 > v39 >> 3 )
        {
LABEL_121:
          *(_DWORD *)(v37 + 16) = v95;
          v96 = v106.m128i_i64[0];
          if ( *(_QWORD *)v106.m128i_i64[0] != -4096 )
            --*(_DWORD *)(v37 + 20);
          v97 = *v32;
          v46 = v96 + 8;
          *(_DWORD *)(v96 + 32) = 0;
          v45 = 0;
          *(_QWORD *)(v96 + 8) = 0;
          v44 = 0;
          *(_QWORD *)v96 = v97;
          *(_QWORD *)(v96 + 16) = 0;
          *(_QWORD *)(v96 + 24) = 0;
LABEL_45:
          if ( v38 )
          {
            v47 = v38[5];
            v104[0] = v47;
            if ( v45 )
              goto LABEL_47;
          }
          else
          {
            v47 = *(_QWORD *)(a1[1][410] + 80);
            if ( v47 )
              v47 -= 24;
            v104[0] = v47;
            if ( v45 )
            {
LABEL_47:
              v48 = (v45 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
              v49 = (__int64 *)(v44 + 24LL * v48);
              v50 = *v49;
              if ( v47 == *v49 )
                goto LABEL_48;
              v88 = 1;
              v89 = 0;
              while ( v50 != -4096 )
              {
                if ( v89 || v50 != -8192 )
                  v49 = (__int64 *)v89;
                v48 = (v45 - 1) & (v88 + v48);
                v50 = *(_QWORD *)(v44 + 24LL * v48);
                if ( v47 == v50 )
                  goto LABEL_48;
                ++v88;
                v89 = (__int64)v49;
                v49 = (__int64 *)(v44 + 24LL * v48);
              }
              if ( !v89 )
                v89 = (__int64)v49;
              v106.m128i_i64[0] = v89;
              v90 = *(_DWORD *)(v46 + 16);
              ++*(_QWORD *)v46;
              v68 = v90 + 1;
              if ( 4 * (v90 + 1) < 3 * v45 )
              {
                if ( v45 - *(_DWORD *)(v46 + 20) - v68 > v45 >> 3 )
                  goto LABEL_112;
                goto LABEL_73;
              }
LABEL_72:
              v45 *= 2;
LABEL_73:
              sub_2B59A70(v46, v45);
              sub_2B40900(v46, v104, &v106);
              v68 = *(_DWORD *)(v46 + 16) + 1;
LABEL_112:
              *(_DWORD *)(v46 + 16) = v68;
              v91 = (__int64 *)v106.m128i_i64[0];
              if ( *(_QWORD *)v106.m128i_i64[0] != -4096 )
                --*(_DWORD *)(v46 + 20);
              v92 = v104[0];
              v91[1] = (__int64)v22;
              v91[2] = (__int64)v27;
              *v91 = v92;
              goto LABEL_48;
            }
          }
          v106.m128i_i64[0] = 0;
          ++*(_QWORD *)v46;
          goto LABEL_72;
        }
        v101 = v32;
LABEL_126:
        sub_2B597F0(v37, v39);
        sub_2B40840(v37, v101, &v106);
        v32 = v101;
        v95 = *(_DWORD *)(v37 + 16) + 1;
        goto LABEL_121;
      }
    }
    else
    {
      v106.m128i_i64[0] = 0;
      ++*(_QWORD *)v37;
    }
    v101 = v32;
    v39 *= 2;
    goto LABEL_126;
  }
  v51 = (__int64)a1[6];
  v53 = *(_DWORD *)(v51 + 24);
  if ( !v53 )
  {
    v106.m128i_i64[0] = 0;
    ++*(_QWORD *)v51;
LABEL_83:
    v53 *= 2;
    goto LABEL_84;
  }
  v54 = *(_QWORD *)(v51 + 8);
  v55 = (v53 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v56 = (_BYTE **)(v54 + 16LL * v55);
  v27 = *v56;
  if ( (_BYTE *)a2 == *v56 )
    return v27;
  v57 = 1;
  v58 = 0;
  while ( v27 != (_BYTE *)-4096LL )
  {
    if ( v27 != (_BYTE *)-8192LL || v58 )
      v56 = (_BYTE **)v58;
    v55 = (v53 - 1) & (v57 + v55);
    v27 = *(_BYTE **)(v54 + 16LL * v55);
    if ( (_BYTE *)a2 == v27 )
      return v27;
    ++v57;
    v58 = (__int64)v56;
    v56 = (_BYTE **)(v54 + 16LL * v55);
  }
  if ( !v58 )
    v58 = (__int64)v56;
  v106.m128i_i64[0] = v58;
  v59 = *(_DWORD *)(v51 + 16);
  ++*(_QWORD *)v51;
  v60 = v59 + 1;
  if ( 4 * (v59 + 1) >= 3 * v53 )
    goto LABEL_83;
  if ( v53 - *(_DWORD *)(v51 + 20) - v60 <= v53 >> 3 )
  {
LABEL_84:
    sub_2B59C70(v51, v53);
    sub_2B409C0(v51, (__int64 *)&v102, &v106);
    v60 = *(_DWORD *)(v51 + 16) + 1;
  }
  *(_DWORD *)(v51 + 16) = v60;
  v61 = (_QWORD *)v106.m128i_i64[0];
  if ( *(_QWORD *)v106.m128i_i64[0] != -4096 )
    --*(_DWORD *)(v51 + 20);
  v62 = v102;
  v61[1] = v4;
  *v61 = v62;
  return v102;
}
