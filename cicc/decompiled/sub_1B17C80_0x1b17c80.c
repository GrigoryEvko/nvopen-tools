// Function: sub_1B17C80
// Address: 0x1b17c80
//
void __fastcall sub_1B17C80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  const __m128i *v4; // r15
  __int64 v6; // r12
  unsigned __int64 v7; // rax
  _QWORD *v8; // r13
  __int64 *v9; // rax
  unsigned __int8 *v10; // rsi
  __int64 v11; // rax
  _QWORD *v12; // rax
  _QWORD *v13; // rbx
  unsigned __int64 *v14; // r14
  __int64 v15; // rax
  unsigned __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rdx
  unsigned __int8 *v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // r14
  __int64 v24; // rdx
  unsigned int v25; // r13d
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  unsigned __int8 *v28; // rsi
  unsigned __int8 *v29; // rsi
  _QWORD *v30; // rax
  _QWORD *v31; // rbx
  unsigned __int64 *v32; // r13
  __int64 v33; // rax
  unsigned __int64 v34; // rcx
  __int64 v35; // rsi
  unsigned __int8 *v36; // rsi
  _QWORD *v37; // rax
  __int64 v38; // rbx
  const __m128i *v39; // r14
  __int64 *v40; // r15
  __int64 v41; // rsi
  _QWORD *v42; // rdx
  _QWORD *v43; // rax
  _QWORD *v44; // r12
  __int64 v45; // rax
  __int64 *v46; // r13
  _QWORD *v47; // rax
  __int64 v48; // rdx
  unsigned __int64 v49; // rax
  __int64 v50; // rax
  __int64 *v51; // rbx
  __int64 *v52; // r12
  __int64 v53; // rdi
  __int64 *v54; // rbx
  __int64 *v55; // r12
  __int64 v56; // rdi
  __int64 *v57; // rbx
  __int64 *v58; // r12
  __int64 *v59; // rsi
  __int64 *v60; // rdx
  __int64 v61; // r8
  __int64 *v62; // r9
  __int64 *v63; // rax
  __int64 *v64; // rdi
  __int64 *v65; // r9
  __int64 v66; // rdx
  __int64 v67; // rsi
  _QWORD *v68; // rdx
  __int64 *v69; // r12
  __int64 *v70; // r15
  __int64 v71; // r14
  unsigned int v72; // edx
  __int64 v73; // rcx
  int v74; // esi
  unsigned int v75; // eax
  __int64 *v76; // r13
  __int64 v77; // rdi
  __int64 *v78; // rbx
  char *v79; // rdx
  char *v80; // rdi
  __int64 v81; // rax
  __int64 v82; // rsi
  char *v83; // rax
  _QWORD *v84; // rax
  __int64 v85; // rax
  __int64 *v86; // rax
  _QWORD *v87; // rdx
  __int64 v88; // rdx
  __int64 v89; // rsi
  __int64 v90; // [rsp+8h] [rbp-148h]
  __int64 v92; // [rsp+20h] [rbp-130h]
  __int64 v93; // [rsp+28h] [rbp-128h]
  __int64 v94; // [rsp+30h] [rbp-120h]
  __int64 v95; // [rsp+30h] [rbp-120h]
  __int64 v96; // [rsp+38h] [rbp-118h]
  __int64 v98; // [rsp+48h] [rbp-108h]
  const __m128i *v99; // [rsp+48h] [rbp-108h]
  unsigned __int8 *v100; // [rsp+58h] [rbp-F8h] BYREF
  unsigned __int8 *v101; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v102; // [rsp+68h] [rbp-E8h]
  unsigned __int64 *v103; // [rsp+70h] [rbp-E0h]
  __int64 *v104; // [rsp+78h] [rbp-D8h]
  __int64 v105; // [rsp+80h] [rbp-D0h]
  int v106; // [rsp+88h] [rbp-C8h]
  __int64 v107; // [rsp+90h] [rbp-C0h]
  __int64 v108; // [rsp+98h] [rbp-B8h]
  unsigned __int8 *v109; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 *v110; // [rsp+B8h] [rbp-98h]
  __int64 *v111; // [rsp+C0h] [rbp-90h]
  __int64 v112; // [rsp+C8h] [rbp-88h]
  int v113; // [rsp+D0h] [rbp-80h]
  _BYTE v114[120]; // [rsp+D8h] [rbp-78h] BYREF

  v4 = (const __m128i *)a1;
  v6 = sub_13FC520(a1);
  if ( a3 )
    sub_1465150(a3, a1);
  v98 = sub_13FA560(a1);
  v7 = sub_157EBA0(v6);
  v8 = (_QWORD *)v7;
  if ( *(_BYTE *)(v7 + 16) != 26 )
  {
    v101 = 0;
    v103 = 0;
    v104 = (__int64 *)sub_16498A0(0);
    v105 = 0;
    v106 = 0;
    v107 = 0;
    v108 = 0;
    v102 = 0;
    BUG();
  }
  v9 = (__int64 *)sub_16498A0(v7);
  v107 = 0;
  v108 = 0;
  v10 = (unsigned __int8 *)v8[6];
  v104 = v9;
  v106 = 0;
  v11 = v8[5];
  v101 = 0;
  v102 = v11;
  v103 = v8 + 3;
  v105 = 0;
  v109 = v10;
  if ( v10 )
  {
    sub_1623A60((__int64)&v109, (__int64)v10, 2);
    v101 = v109;
    if ( v109 )
      sub_1623210((__int64)&v109, v109, (__int64)&v101);
  }
  v94 = **(_QWORD **)(a1 + 32);
  v96 = sub_159C540(v104);
  LOWORD(v111) = 257;
  v12 = sub_1648A60(56, 3u);
  v13 = v12;
  if ( v12 )
    sub_15F83E0((__int64)v12, v94, v98, v96, 0);
  if ( v102 )
  {
    v14 = v103;
    sub_157E9D0(v102 + 40, (__int64)v13);
    v15 = v13[3];
    v16 = *v14;
    v13[4] = v14;
    v16 &= 0xFFFFFFFFFFFFFFF8LL;
    v13[3] = v16 | v15 & 7;
    *(_QWORD *)(v16 + 8) = v13 + 3;
    *v14 = *v14 & 7 | (unsigned __int64)(v13 + 3);
  }
  sub_164B780((__int64)v13, (__int64 *)&v109);
  if ( v101 )
  {
    v100 = v101;
    sub_1623A60((__int64)&v100, (__int64)v101, 2);
    v17 = v13[6];
    v18 = (__int64)(v13 + 6);
    if ( v17 )
    {
      sub_161E7C0((__int64)(v13 + 6), v17);
      v18 = (__int64)(v13 + 6);
    }
    v19 = v100;
    v13[6] = v100;
    if ( v19 )
      sub_1623210((__int64)&v100, v19, v18);
  }
  sub_15F20C0(v8);
  v20 = sub_157F280(v98);
  v22 = v21;
  v23 = v20;
  while ( v22 != v23 )
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
        v24 = *(_QWORD *)(v23 - 8);
      else
        v24 = v23 - 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF);
      *(_QWORD *)(v24 + 24LL * *(unsigned int *)(v23 + 56) + 8) = v6;
      v25 = (*(_DWORD *)(v23 + 20) & 0xFFFFFFF) - 1;
      if ( (*(_DWORD *)(v23 + 20) & 0xFFFFFFF) != 1 )
      {
        do
          sub_15F5350(v23, v25--, 0);
        while ( v25 );
      }
      v26 = *(_QWORD *)(v23 + 32);
      if ( !v26 )
        BUG();
      v23 = 0;
      if ( *(_BYTE *)(v26 - 8) != 77 )
        break;
      v23 = v26 - 24;
      if ( v22 == v26 - 24 )
        goto LABEL_25;
    }
  }
LABEL_25:
  v27 = sub_157EBA0(v6);
  v28 = *(unsigned __int8 **)(v27 + 48);
  v102 = *(_QWORD *)(v27 + 40);
  v103 = (unsigned __int64 *)(v27 + 24);
  v109 = v28;
  if ( v28 )
  {
    sub_1623A60((__int64)&v109, (__int64)v28, 2);
    v29 = v101;
    if ( !v101 )
      goto LABEL_28;
  }
  else
  {
    v29 = v101;
    if ( !v101 )
      goto LABEL_30;
  }
  sub_161E7C0((__int64)&v101, (__int64)v29);
LABEL_28:
  v101 = v109;
  if ( v109 )
    sub_1623210((__int64)&v109, v109, (__int64)&v101);
LABEL_30:
  LOWORD(v111) = 257;
  v30 = sub_1648A60(56, 1u);
  v31 = v30;
  if ( v30 )
    sub_15F8320((__int64)v30, v98, 0);
  if ( v102 )
  {
    v32 = v103;
    sub_157E9D0(v102 + 40, (__int64)v31);
    v33 = v31[3];
    v34 = *v32;
    v31[4] = v32;
    v34 &= 0xFFFFFFFFFFFFFFF8LL;
    v31[3] = v34 | v33 & 7;
    *(_QWORD *)(v34 + 8) = v31 + 3;
    *v32 = *v32 & 7 | (unsigned __int64)(v31 + 3);
  }
  sub_164B780((__int64)v31, (__int64 *)&v109);
  if ( v101 )
  {
    v100 = v101;
    sub_1623A60((__int64)&v100, (__int64)v101, 2);
    v35 = v31[6];
    if ( v35 )
      sub_161E7C0((__int64)(v31 + 6), v35);
    v36 = v100;
    v31[6] = v100;
    if ( v36 )
      sub_1623210((__int64)&v100, v36, (__int64)(v31 + 6));
  }
  v37 = (_QWORD *)sub_157EBA0(v6);
  sub_15F20C0(v37);
  if ( a2 )
  {
    sub_15D7E60(a2, v6, v98);
    sub_15D4360(a2, v6, **(_QWORD **)(a1 + 32));
  }
  v90 = *(_QWORD *)(a1 + 40);
  if ( v90 == *(_QWORD *)(a1 + 32) )
    goto LABEL_167;
  v92 = *(_QWORD *)(a1 + 32);
  do
  {
    v93 = *(_QWORD *)v92 + 40LL;
    v95 = *(_QWORD *)(*(_QWORD *)v92 + 48LL);
    if ( v93 != v95 )
    {
      while ( 1 )
      {
        if ( !v95 )
          BUG();
        v38 = sub_1599EF0(*(__int64 ***)(v95 - 24));
        if ( *(_QWORD *)(v95 - 16) )
          break;
LABEL_62:
        v95 = *(_QWORD *)(v95 + 8);
        if ( v93 == v95 )
          goto LABEL_63;
      }
      v39 = v4;
      v40 = *(__int64 **)(v95 - 16);
      while ( 1 )
      {
        v46 = v40;
        v40 = (__int64 *)v40[1];
        v47 = sub_1648700((__int64)v46);
        if ( *((_BYTE *)v47 + 16) <= 0x17u )
          goto LABEL_54;
        v41 = v47[5];
        v42 = (_QWORD *)v39[4].m128i_i64[1];
        v43 = (_QWORD *)v39[4].m128i_i64[0];
        if ( v42 == v43 )
        {
          v44 = &v43[v39[5].m128i_u32[1]];
          if ( v43 == v44 )
          {
            v68 = (_QWORD *)v39[4].m128i_i64[0];
          }
          else
          {
            do
            {
              if ( v41 == *v43 )
                break;
              ++v43;
            }
            while ( v44 != v43 );
            v68 = v44;
          }
          goto LABEL_100;
        }
        v44 = &v42[v39[5].m128i_u32[0]];
        v43 = sub_16CC9F0(a1 + 56, v41);
        if ( v41 == *v43 )
          break;
        v45 = v39[4].m128i_i64[1];
        if ( v45 == v39[4].m128i_i64[0] )
        {
          v43 = (_QWORD *)(v45 + 8LL * v39[5].m128i_u32[1]);
          v68 = v43;
LABEL_100:
          while ( v68 != v43 && *v43 >= 0xFFFFFFFFFFFFFFFELL )
            ++v43;
          goto LABEL_51;
        }
        v43 = (_QWORD *)(v45 + 8LL * v39[5].m128i_u32[0]);
LABEL_51:
        if ( v43 == v44 )
        {
LABEL_54:
          if ( *v46 )
          {
            v48 = v46[1];
            v49 = v46[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v49 = v48;
            if ( v48 )
              *(_QWORD *)(v48 + 16) = *(_QWORD *)(v48 + 16) & 3LL | v49;
          }
          *v46 = v38;
          if ( !v38 )
            goto LABEL_52;
          v50 = *(_QWORD *)(v38 + 8);
          v46[1] = v50;
          if ( v50 )
            *(_QWORD *)(v50 + 16) = (unsigned __int64)(v46 + 1) | *(_QWORD *)(v50 + 16) & 3LL;
          v46[2] = (v38 + 8) | v46[2] & 3;
          *(_QWORD *)(v38 + 8) = v46;
          if ( !v40 )
          {
LABEL_61:
            v4 = v39;
            goto LABEL_62;
          }
        }
        else
        {
LABEL_52:
          if ( !v40 )
            goto LABEL_61;
        }
      }
      v66 = v39[4].m128i_i64[1];
      if ( v66 == v39[4].m128i_i64[0] )
        v67 = v39[5].m128i_u32[1];
      else
        v67 = v39[5].m128i_u32[0];
      v68 = (_QWORD *)(v66 + 8 * v67);
      goto LABEL_100;
    }
LABEL_63:
    v92 += 8;
  }
  while ( v90 != v92 );
  v51 = (__int64 *)v4[2].m128i_i64[0];
  v52 = (__int64 *)v4[2].m128i_i64[1];
  if ( v51 == v52 )
  {
LABEL_167:
    if ( a4 )
      goto LABEL_69;
    goto LABEL_89;
  }
  do
  {
    v53 = *v51++;
    sub_157EE90(v53);
  }
  while ( v52 != v51 );
  if ( !a4 )
    goto LABEL_89;
  v54 = (__int64 *)v4[2].m128i_i64[0];
  v55 = (__int64 *)v4[2].m128i_i64[1];
  while ( v54 != v55 )
  {
    v56 = *v54++;
    sub_157F980(v56);
  }
LABEL_69:
  v57 = (__int64 *)v4[2].m128i_i64[0];
  v58 = (__int64 *)v4[2].m128i_i64[1];
  v59 = (__int64 *)v114;
  v109 = 0;
  v110 = (__int64 *)v114;
  v111 = (__int64 *)v114;
  v112 = 8;
  v113 = 0;
  if ( v57 == v58 )
    goto LABEL_87;
  v60 = (__int64 *)v114;
  do
  {
LABEL_73:
    v61 = *v57;
    if ( v59 != v60 )
    {
LABEL_71:
      sub_16CCBA0((__int64)&v109, *v57);
      v60 = v111;
      v59 = v110;
      goto LABEL_72;
    }
    v62 = &v59[HIDWORD(v112)];
    if ( v59 == v62 )
    {
LABEL_164:
      if ( HIDWORD(v112) >= (unsigned int)v112 )
        goto LABEL_71;
      ++HIDWORD(v112);
      *v62 = v61;
      v59 = v110;
      ++v109;
      v60 = v111;
    }
    else
    {
      v63 = v59;
      v64 = 0;
      while ( v61 != *v63 )
      {
        if ( *v63 == -2 )
          v64 = v63;
        if ( v62 == ++v63 )
        {
          if ( !v64 )
            goto LABEL_164;
          ++v57;
          *v64 = v61;
          v60 = v111;
          --v113;
          v59 = v110;
          ++v109;
          if ( v58 != v57 )
            goto LABEL_73;
          goto LABEL_82;
        }
      }
    }
LABEL_72:
    ++v57;
  }
  while ( v58 != v57 );
LABEL_82:
  if ( v59 == v60 )
    v65 = &v60[HIDWORD(v112)];
  else
    v65 = &v60[(unsigned int)v112];
  while ( 1 )
  {
    if ( v65 == v60 )
      goto LABEL_87;
    if ( (unsigned __int64)*v60 < 0xFFFFFFFFFFFFFFFELL )
      break;
    ++v60;
  }
  if ( v65 != v60 )
  {
    v99 = v4;
    v69 = v65;
    v70 = v60;
    v71 = *v60;
    while ( 1 )
    {
      v72 = *(_DWORD *)(a4 + 24);
      if ( v72 )
      {
        v73 = *(_QWORD *)(a4 + 8);
        v74 = 1;
        v75 = (v72 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
        v76 = (__int64 *)(v73 + 16LL * v75);
        v77 = *v76;
        if ( v71 == *v76 )
        {
LABEL_110:
          if ( v76 != (__int64 *)(v73 + 16LL * v72) )
          {
            v78 = (__int64 *)v76[1];
            if ( v78 )
            {
              while ( 1 )
              {
                v79 = (char *)v78[5];
                v80 = (char *)v78[4];
                v81 = (v79 - v80) >> 5;
                v82 = (v79 - v80) >> 3;
                if ( v81 <= 0 )
                  break;
                v83 = &v80[32 * v81];
                while ( v71 != *(_QWORD *)v80 )
                {
                  if ( v71 == *((_QWORD *)v80 + 1) )
                  {
                    v80 += 8;
                    break;
                  }
                  if ( v71 == *((_QWORD *)v80 + 2) )
                  {
                    v80 += 16;
                    break;
                  }
                  if ( v71 == *((_QWORD *)v80 + 3) )
                  {
                    v80 += 24;
                    break;
                  }
                  v80 += 32;
                  if ( v80 == v83 )
                  {
                    v82 = (v79 - v80) >> 3;
                    goto LABEL_141;
                  }
                }
LABEL_119:
                if ( v80 + 8 != v79 )
                {
                  memmove(v80, v80 + 8, v79 - (v80 + 8));
                  v79 = (char *)v78[5];
                }
                v84 = (_QWORD *)v78[8];
                v78[5] = (__int64)(v79 - 8);
                if ( (_QWORD *)v78[9] == v84 )
                {
                  v87 = &v84[*((unsigned int *)v78 + 21)];
                  if ( v84 == v87 )
                  {
LABEL_139:
                    v84 = v87;
                  }
                  else
                  {
                    while ( v71 != *v84 )
                    {
                      if ( v87 == ++v84 )
                        goto LABEL_139;
                    }
                  }
                  goto LABEL_134;
                }
                v84 = sub_16CC9F0((__int64)(v78 + 7), v71);
                if ( v71 == *v84 )
                {
                  v88 = v78[9];
                  if ( v88 == v78[8] )
                    v89 = *((unsigned int *)v78 + 21);
                  else
                    v89 = *((unsigned int *)v78 + 20);
                  v87 = (_QWORD *)(v88 + 8 * v89);
LABEL_134:
                  if ( v87 != v84 )
                  {
                    *v84 = -2;
                    ++*((_DWORD *)v78 + 22);
                  }
                  goto LABEL_124;
                }
                v85 = v78[9];
                if ( v85 == v78[8] )
                {
                  v84 = (_QWORD *)(v85 + 8LL * *((unsigned int *)v78 + 21));
                  v87 = v84;
                  goto LABEL_134;
                }
LABEL_124:
                v78 = (__int64 *)*v78;
                if ( !v78 )
                  goto LABEL_125;
              }
LABEL_141:
              if ( v82 != 2 )
              {
                if ( v82 != 3 )
                {
                  if ( v82 != 1 )
                  {
                    v80 = (char *)v78[5];
                    goto LABEL_119;
                  }
LABEL_155:
                  if ( v71 != *(_QWORD *)v80 )
                    v80 = (char *)v78[5];
                  goto LABEL_119;
                }
                if ( v71 == *(_QWORD *)v80 )
                  goto LABEL_119;
                v80 += 8;
              }
              if ( v71 == *(_QWORD *)v80 )
                goto LABEL_119;
              v80 += 8;
              goto LABEL_155;
            }
LABEL_125:
            *v76 = -16;
            --*(_DWORD *)(a4 + 16);
            ++*(_DWORD *)(a4 + 20);
          }
        }
        else
        {
          while ( v77 != -8 )
          {
            v75 = (v72 - 1) & (v74 + v75);
            v76 = (__int64 *)(v73 + 16LL * v75);
            v77 = *v76;
            if ( v71 == *v76 )
              goto LABEL_110;
            ++v74;
          }
        }
      }
      v86 = v70 + 1;
      if ( v70 + 1 != v69 )
      {
        while ( 1 )
        {
          v71 = *v86;
          v70 = v86;
          if ( (unsigned __int64)*v86 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v69 == ++v86 )
            goto LABEL_129;
        }
        if ( v69 != v86 )
          continue;
      }
LABEL_129:
      v4 = v99;
      break;
    }
  }
LABEL_87:
  sub_1401B00(a4, v4);
  if ( v111 != v110 )
    _libc_free((unsigned __int64)v111);
LABEL_89:
  if ( v101 )
    sub_161E7C0((__int64)&v101, (__int64)v101);
}
