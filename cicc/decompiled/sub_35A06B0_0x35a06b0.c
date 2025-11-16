// Function: sub_35A06B0
// Address: 0x35a06b0
//
__int64 __fastcall sub_35A06B0(__int64 *a1)
{
  __int64 *v1; // r13
  __int64 *v2; // r15
  __int64 *v3; // r14
  void *v4; // r12
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rax
  void *i; // rsi
  __int64 v9; // rax
  void *v10; // r14
  __int64 v11; // rdi
  __int64 v12; // rbx
  int v13; // eax
  __int64 v14; // rbx
  __int64 v15; // r14
  int v16; // r14d
  unsigned __int64 v17; // rbx
  __int64 v18; // r12
  int v19; // r15d
  __int64 v20; // r12
  __int64 result; // rax
  __int64 v22; // r14
  unsigned int v23; // eax
  __int64 v24; // r12
  __int64 v25; // r14
  int v26; // r8d
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // rdx
  int v32; // esi
  int v33; // r12d
  unsigned __int64 v34; // r8
  unsigned int v35; // r9d
  __int64 v36; // rsi
  __int64 v37; // r8
  __int32 v38; // eax
  __int64 v39; // r8
  __int64 v40; // r9
  unsigned __int64 v41; // r10
  char *v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // rbx
  int v45; // edx
  int v46; // eax
  __int64 v47; // r9
  int v48; // edx
  __int64 v49; // rcx
  int v50; // ebx
  int v51; // r15d
  unsigned __int64 v52; // r8
  __int64 v53; // rdx
  _QWORD *v54; // rax
  _BYTE *v55; // rdi
  __int64 v56; // r9
  char *v57; // r8
  int v58; // esi
  __int64 v59; // r12
  int v60; // r13d
  char *v61; // rbx
  unsigned __int64 v62; // rdx
  int v63; // eax
  __int32 v64; // eax
  __int64 v65; // rdi
  __int32 v66; // r8d
  __int64 v67; // rax
  __int64 v68; // rcx
  _QWORD *v69; // rax
  __int64 v70; // rdx
  __int64 *v71; // r8
  __int64 v72; // rax
  __int64 v73; // rdi
  __int64 *v74; // r12
  __int64 v75; // rax
  __int64 v76; // rdi
  __int64 v77; // r12
  __int64 v78; // rdi
  bool v79; // zf
  __int64 v80; // rax
  int v81; // esi
  int v82; // edx
  unsigned int v83; // esi
  __int64 v84; // rdx
  char *v85; // rdi
  unsigned int v86; // r15d
  __int64 v87; // rdx
  __int64 v88; // rbx
  int v89; // r15d
  int v90; // ebx
  int v91; // esi
  int v92; // eax
  __int64 v93; // [rsp+10h] [rbp-130h]
  __int64 v94; // [rsp+18h] [rbp-128h]
  __int64 *v95; // [rsp+18h] [rbp-128h]
  int v96; // [rsp+2Ch] [rbp-114h]
  unsigned __int8 v97; // [rsp+2Ch] [rbp-114h]
  int v98; // [rsp+2Ch] [rbp-114h]
  __int64 v99; // [rsp+30h] [rbp-110h]
  __int64 v100; // [rsp+38h] [rbp-108h]
  __int64 v101; // [rsp+50h] [rbp-F0h]
  __int64 *v102; // [rsp+58h] [rbp-E8h]
  __int64 v103; // [rsp+58h] [rbp-E8h]
  __int64 v104; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v105; // [rsp+68h] [rbp-D8h] BYREF
  __int64 v106; // [rsp+70h] [rbp-D0h]
  __int64 v107; // [rsp+78h] [rbp-C8h]
  __int64 v108[2]; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v109[4]; // [rsp+90h] [rbp-B0h] BYREF
  __m128i v110; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v111; // [rsp+C0h] [rbp-80h]
  __int64 v112; // [rsp+C8h] [rbp-78h]
  void *dest; // [rsp+E0h] [rbp-60h] BYREF
  __int64 j; // [rsp+E8h] [rbp-58h]
  _BYTE v115[80]; // [rsp+F0h] [rbp-50h] BYREF

  v1 = a1;
  v2 = (__int64 *)sub_2E313E0(a1[1]);
  v102 = *(__int64 **)(*a1 + 16);
  if ( *(__int64 **)(*a1 + 8) == v102 )
  {
    v4 = 0;
  }
  else
  {
    v3 = *(__int64 **)(*a1 + 8);
    v4 = 0;
    do
    {
      v5 = *v3;
      if ( *(_WORD *)(*v3 + 68) != 68 && *(_WORD *)(*v3 + 68) )
      {
        if ( *(_QWORD *)(v5 + 24) )
          sub_2E88DB0((_QWORD *)*v3);
        sub_2E31040((__int64 *)(a1[1] + 40), v5);
        v6 = *v2;
        v7 = *(_QWORD *)v5;
        *(_QWORD *)(v5 + 8) = v2;
        v6 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v5 = v6 | v7 & 7;
        *(_QWORD *)(v6 + 8) = v5;
        *v2 = v5 | *v2 & 7;
        if ( !v4 )
          v4 = (void *)v5;
      }
      ++v3;
    }
    while ( v102 != v3 );
  }
  dest = (void *)sub_2E311E0(a1[1]);
  for ( i = dest; dest != v4; i = dest )
  {
    v9 = a1[6];
    if ( v9 )
      sub_2FAD510(*(_QWORD *)(v9 + 32), (__int64)i);
    v10 = dest;
    sub_2FD79B0((__int64 *)&dest);
    sub_2E88E20((__int64)v10);
  }
  v11 = a1[1];
  v12 = *(_QWORD *)(v11 + 56);
  v100 = v11 + 48;
  v104 = v12;
  if ( v11 + 48 != v12 )
  {
    v103 = v12;
    while ( 1 )
    {
      if ( *(_WORD *)(v103 + 68) == 68 || !*(_WORD *)(v103 + 68) )
        goto LABEL_18;
      v13 = *(_DWORD *)(v103 + 44);
      if ( (v13 & 4) != 0 || (v13 & 8) == 0 )
      {
        if ( (*(_QWORD *)(*(_QWORD *)(v103 + 16) + 24LL) & 0x200LL) == 0 )
        {
LABEL_23:
          v14 = *(_QWORD *)(v103 + 32);
          v15 = v14 + 40LL * (*(_DWORD *)(v103 + 40) & 0xFFFFFF);
          v99 = v15;
          v101 = v14 + 40LL * (unsigned int)sub_2E88FE0(v103);
          if ( v15 == v101 )
            goto LABEL_18;
          while ( 2 )
          {
            if ( *(_BYTE *)v101 )
              goto LABEL_32;
            v16 = *(_DWORD *)(v101 + 8);
            if ( (unsigned int)(v16 - 1) <= 0x3FFFFFFE || (*(_BYTE *)(v101 + 3) & 0x20) != 0 )
              goto LABEL_32;
            v17 = sub_2EBEE90(v1[4], v16);
            if ( !v17 )
              goto LABEL_31;
            v18 = *v1;
            v19 = sub_3598DB0(*v1, v103);
            if ( *(_WORD *)(v17 + 68) != 68 && *(_WORD *)(v17 + 68) )
            {
              if ( *(_QWORD *)(v17 + 24) == v1[1] )
              {
                v89 = v19 - sub_3598DB0(v18, v17);
                if ( v89 )
                {
                  v90 = 0;
                  v91 = v16;
                  do
                  {
                    HIDWORD(dest) = 0;
                    ++v90;
                    v92 = sub_359FDD0((__int64)v1, v91, (unsigned int)dest, 0);
                    v91 = v92;
                  }
                  while ( v89 != v90 );
                  v16 = v92;
                }
              }
LABEL_31:
              sub_2EAB0C0(v101, v16);
              goto LABEL_32;
            }
            v33 = v16;
            v34 = v17;
            v35 = 0;
            dest = v115;
            for ( j = 0x400000000LL; ; v35 = j )
            {
              v36 = v1[1];
              if ( *(_QWORD *)(v34 + 24) != v36 )
                break;
              v33 = sub_3598190(v34, v36);
              v38 = sub_3598140(v37, v36);
              if ( HIDWORD(j) <= v41 )
              {
                v110.m128i_i32[0] = v38;
                v110.m128i_i8[4] = 1;
                v88 = v110.m128i_i64[0];
                if ( HIDWORD(j) < v41 + 1 )
                {
                  sub_C8D5F0((__int64)&dest, v115, v41 + 1, 8u, v39, v40);
                  v41 = (unsigned int)j;
                }
                *((_QWORD *)dest + v41) = v88;
                LODWORD(j) = j + 1;
              }
              else
              {
                v42 = (char *)dest + 8 * v41;
                if ( v42 )
                {
                  *(_DWORD *)v42 = v38;
                  v42[4] = 1;
                  LODWORD(v40) = j;
                }
                LODWORD(j) = v40 + 1;
              }
              v34 = sub_2EBEE90(v1[4], v33);
              if ( *(_WORD *)(v34 + 68) != 68 && *(_WORD *)(v34 + 68) )
              {
                v43 = (unsigned int)j;
                v44 = v34;
                v45 = j;
                goto LABEL_69;
              }
            }
            v44 = v34;
            v45 = v35;
            v43 = v35;
LABEL_69:
            v94 = v43;
            v96 = v45;
            v46 = sub_3598DB0(*v1, v44);
            v48 = v96;
            v106 = 0;
            v49 = v94;
            v50 = v46;
            if ( v46 == -1 )
              goto LABEL_85;
            if ( v19 < v46 )
            {
              v85 = (char *)dest;
              v86 = *((unsigned __int8 *)dest + 4);
              v106 = *(_QWORD *)dest;
              if ( (char *)dest + 8 != (char *)dest + 8 * v94 )
              {
                memmove(dest, (char *)dest + 8, 8 * v94 - 8);
                v48 = j;
                v85 = (char *)dest;
              }
              v87 = (unsigned int)(v48 - 1);
              v56 = v86;
              v57 = &v85[8 * v87];
              LODWORD(j) = v87;
              if ( v57 != v85 )
                break;
              goto LABEL_89;
            }
            v51 = v19 - v46;
            if ( !v51 )
            {
LABEL_85:
              v55 = dest;
              LOBYTE(v56) = 0;
              v57 = (char *)dest + 8 * v94;
              if ( v57 == dest )
                goto LABEL_81;
              break;
            }
            v52 = v51 + v94;
            if ( v96 )
            {
              v107 = *((_QWORD *)dest + v94 - 1);
              v109[0] = v107;
            }
            else
            {
              HIDWORD(v107) = 0;
              v109[0] = (unsigned int)v107;
            }
            v110.m128i_i64[0] = v109[0];
            if ( HIDWORD(j) < v52 )
            {
              sub_C8D5F0((__int64)&dest, v115, v52, 8u, v52, v47);
              v49 = (unsigned int)j;
            }
            v53 = v51;
            v54 = (char *)dest + 8 * v49;
            do
            {
              if ( v54 )
                *v54 = v110.m128i_i64[0];
              ++v54;
              --v53;
            }
            while ( v53 );
            v55 = dest;
            LOBYTE(v56) = 0;
            LODWORD(j) = j + v51;
            v57 = (char *)dest + 8 * (unsigned int)j;
            if ( v57 == dest )
            {
LABEL_81:
              v16 = v33;
LABEL_82:
              if ( v55 == v115 )
                goto LABEL_31;
              _libc_free((unsigned __int64)v55);
              sub_2EAB0C0(v101, v16);
LABEL_32:
              v101 += 40;
              if ( v99 == v101 )
                goto LABEL_18;
              continue;
            }
            break;
          }
          v58 = v33;
          v59 = (__int64)v1;
          v60 = v50;
          v61 = v57;
          do
          {
            v62 = *((_QWORD *)v61 - 1);
            v61 -= 8;
            v97 = v56;
            v63 = sub_359FDD0(
                    v59,
                    v58,
                    v62,
                    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v59 + 32) + 56LL) + 16LL * (v16 & 0x7FFFFFFF))
                  & 0xFFFFFFFFFFFFFFF8LL);
            v56 = v97;
            v58 = v63;
          }
          while ( dest != v61 );
          v50 = v60;
          v1 = (__int64 *)v59;
          v33 = v63;
LABEL_89:
          if ( !(_BYTE)v56 )
          {
            v55 = dest;
            goto LABEL_81;
          }
          v64 = sub_2EC06C0(
                  v1[4],
                  *(_QWORD *)(*(_QWORD *)(v1[4] + 56) + 16LL * (v16 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                  byte_3F871B3,
                  0,
                  (__int64)v57,
                  v56);
          v65 = v1[1];
          v66 = v64;
          v98 = v64;
          v67 = v1[5];
          v105 = 0;
          v68 = *(_QWORD *)(v67 + 8);
          memset(v109, 0, 24);
          v69 = sub_2F2A600(v65, v103, v109, v68, v66);
          v108[1] = v70;
          v108[0] = (__int64)v69;
          v71 = sub_3598AB0(v108, v106, 0, 0);
          v72 = v1[2];
          v73 = v71[1];
          v110.m128i_i8[0] = 4;
          v112 = v72;
          v110.m128i_i32[0] &= 0xFFF000FF;
          v111 = 0;
          v95 = v71;
          sub_2E8EAD0(v73, *v71, &v110);
          v74 = sub_3598AB0(v95, v33, 0, 0);
          v75 = v1[1];
          v76 = v74[1];
          v110.m128i_i8[0] = 4;
          v112 = v75;
          v110.m128i_i32[0] &= 0xFFF000FF;
          v111 = 0;
          sub_2E8EAD0(v76, *v74, &v110);
          v93 = v74[1];
          sub_9C6650(v109);
          sub_9C6650(&v105);
          v77 = *v1;
          v78 = *v1 + 64;
          v108[0] = v93;
          v79 = (unsigned __int8)sub_3546FB0(v78, v108, v109) == 0;
          v80 = v109[0];
          if ( v79 )
          {
            v110.m128i_i64[0] = v109[0];
            v81 = *(_DWORD *)(v77 + 80);
            ++*(_QWORD *)(v77 + 64);
            v82 = v81 + 1;
            v83 = *(_DWORD *)(v77 + 88);
            if ( 4 * v82 >= 3 * v83 )
            {
              v83 *= 2;
            }
            else if ( v83 - *(_DWORD *)(v77 + 84) - v82 > v83 >> 3 )
            {
              goto LABEL_95;
            }
            sub_354C5D0(v78, v83);
            sub_3546FB0(v78, v108, &v110);
            v82 = *(_DWORD *)(v77 + 80) + 1;
            v80 = v110.m128i_i64[0];
LABEL_95:
            *(_DWORD *)(v77 + 80) = v82;
            if ( *(_QWORD *)v80 != -4096 )
              --*(_DWORD *)(v77 + 84);
            v84 = v108[0];
            *(_DWORD *)(v80 + 8) = 0;
            *(_QWORD *)v80 = v84;
          }
          *(_DWORD *)(v80 + 8) = v50;
          v16 = v98;
          v55 = dest;
          goto LABEL_82;
        }
      }
      else if ( !sub_2E88A90(v103, 512, 1) )
      {
        goto LABEL_23;
      }
LABEL_18:
      sub_2FD79B0(&v104);
      v103 = v104;
      if ( v100 == v104 )
      {
        v11 = v1[1];
        break;
      }
    }
  }
  sub_3598750(v11, (_QWORD *)v1[4], v1[6], 0);
  v110.m128i_i64[0] = sub_2E311E0(v1[1]);
  v20 = v110.m128i_i64[0];
  for ( result = v1[1] + 48; v110.m128i_i64[0] != result; result = v1[1] + 48 )
  {
    v22 = *(_QWORD *)(v20 + 32);
    if ( *(_WORD *)(v20 + 68) == 68 || !*(_WORD *)(v20 + 68) )
    {
      v32 = *(_DWORD *)(v22 + 8);
      HIDWORD(dest) = 0;
      sub_359FDD0((__int64)v1, v32, (unsigned int)dest, 0);
    }
    else
    {
      v23 = sub_2E88FE0(v20);
      v24 = *(_QWORD *)(v20 + 32);
      v25 = v22 + 40LL * v23;
      if ( v25 != v24 )
      {
        while ( 1 )
        {
          while ( 1 )
          {
            v26 = *(_DWORD *)(v24 + 8);
            v27 = v1[4];
            v28 = v26 < 0
                ? *(_QWORD *)(*(_QWORD *)(v27 + 56) + 16LL * (v26 & 0x7FFFFFFF) + 8)
                : *(_QWORD *)(*(_QWORD *)(v27 + 304) + 8LL * (unsigned int)v26);
            if ( v28 )
              break;
LABEL_47:
            v24 += 40;
            if ( v24 == v25 )
              goto LABEL_48;
          }
          if ( (*(_BYTE *)(v28 + 3) & 0x10) != 0 )
          {
            while ( 1 )
            {
              v28 = *(_QWORD *)(v28 + 32);
              if ( !v28 )
                break;
              if ( (*(_BYTE *)(v28 + 3) & 0x10) == 0 )
                goto LABEL_43;
            }
            v24 += 40;
            if ( v24 == v25 )
              break;
          }
          else
          {
LABEL_43:
            v29 = *(_QWORD *)(v28 + 16);
            v30 = v1[1];
            if ( v30 == *(_QWORD *)(v29 + 24) )
            {
              while ( 1 )
              {
                v28 = *(_QWORD *)(v28 + 32);
                if ( !v28 )
                  goto LABEL_47;
                if ( (*(_BYTE *)(v28 + 3) & 0x10) == 0 )
                {
                  v31 = *(_QWORD *)(v28 + 16);
                  if ( v31 != v29 )
                  {
                    v29 = *(_QWORD *)(v28 + 16);
                    if ( v30 != *(_QWORD *)(v31 + 24) )
                      break;
                  }
                }
              }
            }
            HIDWORD(dest) = 0;
            v24 += 40;
            sub_359FDD0((__int64)v1, v26, (unsigned int)dest, 0);
            if ( v24 == v25 )
              break;
          }
        }
      }
    }
LABEL_48:
    sub_2FD79B0(v110.m128i_i64);
    v20 = v110.m128i_i64[0];
  }
  return result;
}
