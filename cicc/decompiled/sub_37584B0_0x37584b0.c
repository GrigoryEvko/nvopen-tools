// Function: sub_37584B0
// Address: 0x37584b0
//
void __fastcall sub_37584B0(__int64 *a1, __int64 a2, char a3, __int64 a4, __m128i *a5, __int64 a6)
{
  __int64 *v6; // r13
  unsigned __int64 v7; // r12
  unsigned __int16 *v8; // r15
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 (*v11)(); // rax
  unsigned int v12; // ecx
  unsigned int v13; // edx
  __int64 v14; // rsi
  __int16 v15; // ax
  __int64 v16; // rcx
  __int64 v17; // rsi
  int v18; // eax
  __int64 (*v19)(); // rax
  bool v20; // dl
  unsigned __int8 *v21; // rsi
  _QWORD *v22; // rbx
  _QWORD *v23; // rax
  __int64 v24; // r14
  int v25; // eax
  __int64 v26; // r14
  unsigned int v27; // eax
  unsigned int v28; // ebx
  unsigned __int64 *v29; // rax
  unsigned int v30; // r8d
  unsigned __int16 v31; // ax
  __int64 v32; // rcx
  _QWORD *v33; // rdx
  __int64 v34; // rbx
  __int64 *v35; // r14
  __int64 j; // r8
  unsigned __int64 v37; // r9
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 i; // r14
  signed int v41; // ebx
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // rdx
  int v46; // eax
  unsigned int *v47; // rax
  __int64 v48; // rdx
  __int32 v49; // eax
  unsigned int v50; // eax
  __int64 v51; // rbx
  int v52; // eax
  __int64 v53; // rdx
  __int64 v54; // r14
  unsigned int v55; // r13d
  unsigned int v56; // r12d
  __int64 v57; // rax
  int v58; // r14d
  __int64 v59; // rax
  unsigned __int64 v60; // rdx
  __int64 v61; // rcx
  unsigned __int16 *v62; // rax
  __int64 v63; // r14
  __int64 v64; // r10
  unsigned __int64 v65; // rdx
  __int64 v66; // r8
  unsigned __int16 *v67; // rax
  unsigned __int16 *v68; // r8
  _DWORD *v69; // rdx
  __int64 v70; // r14
  int v71; // eax
  __int64 v72; // rdx
  __int64 v73; // rdi
  __int64 (*v74)(); // rax
  __int64 v75; // rax
  __int64 v76; // rdx
  unsigned __int16 *v77; // rbx
  __m128i *v78; // rax
  unsigned __int16 *v79; // r12
  int v80; // r14d
  unsigned __int16 *v82; // [rsp+10h] [rbp-100h]
  bool v83; // [rsp+20h] [rbp-F0h]
  unsigned int v84; // [rsp+28h] [rbp-E8h]
  unsigned __int16 *v85; // [rsp+28h] [rbp-E8h]
  unsigned int v86; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v87; // [rsp+30h] [rbp-E0h]
  __int16 v88; // [rsp+30h] [rbp-E0h]
  __int64 v89; // [rsp+30h] [rbp-E0h]
  unsigned int v90; // [rsp+30h] [rbp-E0h]
  char v92; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v93; // [rsp+40h] [rbp-D0h]
  __int64 *v94; // [rsp+40h] [rbp-D0h]
  int v95; // [rsp+40h] [rbp-D0h]
  __m128i *v96; // [rsp+40h] [rbp-D0h]
  unsigned __int16 v97; // [rsp+40h] [rbp-D0h]
  unsigned int v98; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v99; // [rsp+48h] [rbp-C8h]
  unsigned int v100; // [rsp+50h] [rbp-C0h]
  unsigned int v101; // [rsp+54h] [rbp-BCh]
  unsigned __int8 *v103; // [rsp+68h] [rbp-A8h] BYREF
  _QWORD *v104; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v105; // [rsp+78h] [rbp-98h]
  __m128i v106; // [rsp+80h] [rbp-90h] BYREF
  __int64 v107; // [rsp+90h] [rbp-80h]
  __int64 v108; // [rsp+98h] [rbp-78h]
  __int64 v109; // [rsp+A0h] [rbp-70h]
  __m128i v110; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v111[10]; // [rsp+C0h] [rbp-50h] BYREF

  v6 = a1;
  v7 = a2;
  v92 = a4;
  v100 = ~*(_DWORD *)(a2 + 24);
  v83 = v100 == 12 || (unsigned int)(-*(_DWORD *)(a2 + 24) - 9) <= 1;
  if ( v83 )
  {
    sub_3757400(a1, a2, a5, a3, (unsigned __int8)a4, a6);
    return;
  }
  if ( v100 == 13 )
  {
    sub_3757DF0(a1, (unsigned __int8 *)a2, a5, a4, (__int64)a5, a6);
    return;
  }
  if ( v100 == 19 )
  {
    sub_3757FE0(a1, (unsigned __int8 *)a2, a5, a3, a4);
    return;
  }
  if ( v100 != 10 )
  {
    v8 = (unsigned __int16 *)(*(_QWORD *)(a1[2] + 8) - 40LL * v100);
    v84 = sub_3751FC0(a2);
    v101 = *((unsigned __int8 *)v8 + 4);
    if ( ((v100 - 26) & 0xFFFFFFFD) != 0 )
    {
      v50 = *((unsigned __int8 *)v8 + 4);
      v82 = 0;
      if ( v100 == 32 )
        v50 = v84;
      v101 = v50;
    }
    else
    {
      v9 = 13;
      if ( v100 == 28 )
      {
        v57 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 40) + 160LL) + 96LL);
        v9 = *(_QWORD *)(v57 + 24);
        if ( *(_DWORD *)(v57 + 32) > 0x40u )
          v9 = *(_QWORD *)v9;
        v101 = v84;
      }
      v82 = 0;
      v10 = a1[4];
      v11 = *(__int64 (**)())(*(_QWORD *)v10 + 2384LL);
      if ( v11 != sub_302E260 )
        v82 = (unsigned __int16 *)((__int64 (__fastcall *)(__int64, __int64))v11)(v10, v9);
    }
    v12 = *(_DWORD *)(v7 + 64);
    v13 = v8[1] - v101;
    while ( 1 )
    {
      if ( !v12 )
      {
        v98 = 0;
        goto LABEL_19;
      }
      v14 = *(_QWORD *)(v7 + 40);
      v15 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v14 + 40LL * (v12 - 1)) + 48LL)
                     + 16LL * *(unsigned int *)(v14 + 40LL * (v12 - 1) + 8));
      if ( v15 != 262 )
        break;
      --v12;
    }
    if ( v15 == 1 )
      --v12;
    v98 = v12;
    if ( v13 < v12 )
    {
      v16 = v14 + 40LL * (v12 - 1);
      v17 = v16 - 40 - 40LL * (~v13 + v98);
      do
      {
        v18 = *(_DWORD *)(*(_QWORD *)v16 + 24LL);
        if ( v18 != 10 && (v18 != 9 || (unsigned int)(*(_DWORD *)(*(_QWORD *)v16 + 96LL) - 1) > 0x3FFFFFFE) )
          break;
        v16 -= 40;
      }
      while ( v17 != v16 );
    }
LABEL_19:
    v19 = *(__int64 (**)())(**(_QWORD **)(*v6 + 8) + 224LL);
    v20 = v19 != sub_23CE3C0 && !(unsigned __int8)v19() && (*((_QWORD *)v8 + 3) & 0x8000000002LL) == 0x8000000002LL;
    if ( v101 < v84 )
      v83 = !v20 && *((_BYTE *)v8 + 9) != 0;
    v21 = *(unsigned __int8 **)(v7 + 80);
    v103 = v21;
    if ( v21 )
    {
      sub_B96E90((__int64)&v103, (__int64)v21, 1);
      v110.m128i_i64[0] = (__int64)v103;
      if ( v103 )
      {
        sub_B976B0((__int64)&v103, v103, (__int64)&v110);
        v103 = 0;
        v110.m128i_i64[1] = 0;
        v22 = (_QWORD *)*v6;
        v111[0] = 0;
        v106.m128i_i64[0] = v110.m128i_i64[0];
        if ( v110.m128i_i64[0] )
          sub_B96E90((__int64)&v106, v110.m128i_i64[0], 1);
        v23 = sub_2E7B380(v22, (__int64)v8, (unsigned __int8 **)&v106, 0);
        goto LABEL_28;
      }
    }
    else
    {
      v110.m128i_i64[0] = 0;
    }
    v22 = (_QWORD *)*v6;
    v110.m128i_i64[1] = 0;
    v111[0] = 0;
    v106.m128i_i64[0] = 0;
    v23 = sub_2E7B380(v22, (__int64)v8, (unsigned __int8 **)&v106, 0);
LABEL_28:
    v24 = (__int64)v23;
    if ( v110.m128i_i64[1] )
      sub_2E882B0((__int64)v23, (__int64)v22, v110.m128i_i64[1]);
    if ( v111[0] )
      sub_2E88680(v24, (__int64)v22, v111[0]);
    if ( v106.m128i_i64[0] )
      sub_B91220((__int64)&v106, v106.m128i_i64[0]);
    v104 = v22;
    v105 = v24;
    if ( v110.m128i_i64[0] )
      sub_B91220((__int64)&v110, v110.m128i_i64[0]);
    if ( v103 )
      sub_B91220((__int64)&v103, (__int64)v103);
    v25 = *(_DWORD *)(v7 + 28);
    v26 = v105;
    if ( (v25 & 0x2000) != 0 )
      *(_DWORD *)(v105 + 44) |= 0x10000u;
    if ( v84 )
    {
      v88 = v25;
      sub_3756EA0(v6, v7, (__int64 *)&v104, v8, a3, v92, a5);
      if ( (v88 & 0x80u) != 0 )
        *(_DWORD *)(v26 + 44) |= 0x40u;
      if ( (v88 & 0x100) != 0 )
        *(_DWORD *)(v26 + 44) |= 0x80u;
      if ( (v88 & 0x20) != 0 )
        *(_DWORD *)(v26 + 44) |= 0x10u;
      if ( (v88 & 0x40) != 0 )
        *(_DWORD *)(v26 + 44) |= 0x20u;
      if ( (v88 & 0x200) != 0 )
        *(_DWORD *)(v26 + 44) |= 0x100u;
      if ( (v88 & 0x400) != 0 )
        *(_DWORD *)(v26 + 44) |= 0x200u;
      if ( (v88 & 0x800) != 0 )
        *(_DWORD *)(v26 + 44) |= 0x400u;
      if ( (v88 & 1) != 0 )
        *(_DWORD *)(v26 + 44) |= 0x800u;
      if ( (v88 & 2) != 0 )
        *(_DWORD *)(v26 + 44) |= 0x1000u;
      if ( (v88 & 4) != 0 )
        *(_DWORD *)(v26 + 44) |= 0x2000u;
      if ( (v88 & 0x1000) != 0 )
        *(_DWORD *)(v26 + 44) |= 0x4000u;
      if ( (v88 & 8) != 0 )
        *(_DWORD *)(v26 + 44) |= 0x80000u;
      if ( (v88 & 0x4000) != 0 )
        *(_DWORD *)(v26 + 44) |= 0x200000u;
    }
    v27 = 0;
    if ( v101 > v84 )
      v27 = v101 - v84;
    if ( v27 != v98 )
    {
      v86 = v101 - v27;
      v28 = v27;
      do
      {
        v29 = (unsigned __int64 *)(*(_QWORD *)(v7 + 40) + 40LL * v28);
        v30 = v28 + v86;
        ++v28;
        sub_3752760(v6, (__int64 *)&v104, *v29, v29[1], v30, (__int64)v8, (__int64)a5, 0, a3, v92);
      }
      while ( v28 != v98 );
    }
    if ( v82 )
    {
      v31 = *v82;
      if ( *v82 )
      {
        v93 = v7;
        LODWORD(v7) = 0;
        do
        {
          v110.m128i_i32[2] = v31;
          memset(v111, 0, 24);
          v110.m128i_i64[0] = 0x430000000LL;
          sub_2E8EAD0(v105, (__int64)v104, &v110);
          v7 = (unsigned int)(v7 + 1);
          v31 = v82[v7];
        }
        while ( v31 );
        v7 = v93;
      }
    }
    v32 = *(int *)(v7 + 104);
    if ( (_DWORD)v32 )
    {
      if ( (_DWORD)v32 == 1 )
      {
        v33 = (_QWORD *)(v7 + 96);
        v32 = 1;
      }
      else
      {
        v33 = (_QWORD *)(*(_QWORD *)(v7 + 96) & 0xFFFFFFFFFFFFFFF8LL);
      }
    }
    else
    {
      v32 = 0;
      v33 = 0;
    }
    sub_2E86A90(v105, (__int64)v104, v33, v32);
    sub_2E88490(v105, *v6, *(_DWORD *)(v7 + 92));
    v34 = v105;
    v35 = (__int64 *)v6[6];
    sub_2E31040((__int64 *)(v6[5] + 40), v105);
    v38 = *v35;
    v39 = *(_QWORD *)v34;
    *(_QWORD *)(v34 + 8) = v35;
    v38 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v34 = v38 | v39 & 7;
    *(_QWORD *)(v38 + 8) = v34;
    *v35 = *v35 & 7 | v34;
    v110.m128i_i64[0] = (__int64)v111;
    v110.m128i_i64[1] = 0x800000000LL;
    if ( v83 && v101 < v84 )
    {
      for ( i = 0; ; ++i )
      {
        v41 = v8[20 * *v8 + 20 + *((unsigned __int8 *)v8 + 8) + *((unsigned int *)v8 + 3) + i];
        if ( (unsigned __int8)sub_33CF8A0(v7, v101 + (unsigned int)i) )
        {
          v42 = v110.m128i_u32[2];
          v43 = v110.m128i_u32[2] + 1LL;
          if ( v43 > v110.m128i_u32[3] )
          {
            sub_C8D5F0((__int64)&v110, v111, v43, 4u, j, v37);
            v42 = v110.m128i_u32[2];
          }
          *(_DWORD *)(v110.m128i_i64[0] + 4 * v42) = v41;
          ++v110.m128i_i32[2];
          sub_37553D0((__int64)v6, (unsigned __int8 *)v7, v101 + i, a3, v41, a5);
        }
        if ( v84 - 1 - v101 == i )
          break;
      }
    }
    if ( *(_WORD *)(*(_QWORD *)(v7 + 48) + 16LL * (unsigned int)(*(_DWORD *)(v7 + 68) - 1)) == 262 )
    {
      v44 = *(_QWORD *)(v7 + 56);
      if ( v44 )
      {
        while ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v44 + 48LL) + 16LL * *(unsigned int *)(v44 + 8)) != 262 )
        {
          v44 = *(_QWORD *)(v44 + 32);
          if ( !v44 )
            goto LABEL_66;
        }
        v51 = *(_QWORD *)(v44 + 16);
        if ( v51 )
        {
          v52 = *(_DWORD *)(v51 + 24);
          if ( v52 == 50 )
            goto LABEL_114;
LABEL_94:
          if ( v52 != 49 )
          {
            v61 = v110.m128i_u32[2];
            v62 = (unsigned __int16 *)(*(_QWORD *)(v6[2] + 8) - 40LL * (unsigned int)~v52);
            v63 = *((unsigned __int8 *)v62 + 8);
            v37 = *v62;
            v64 = *((unsigned int *)v62 + 3);
            v65 = v110.m128i_u32[2] + v63;
            v66 = v63;
            if ( v65 > v110.m128i_u32[3] )
            {
              v85 = v62;
              v90 = *((_DWORD *)v62 + 3);
              v97 = *v62;
              sub_C8D5F0((__int64)&v110, v111, v65, 4u, v66 * 2, v37);
              v61 = v110.m128i_u32[2];
              v66 = v63;
              v62 = v85;
              v64 = v90;
              LOWORD(v37) = v97;
            }
            v37 = (unsigned __int16)v37;
            v67 = &v62[20 * (unsigned __int16)v37 + 20 + v64];
            v68 = &v67[v66];
            v69 = (_DWORD *)(v110.m128i_i64[0] + 4 * v61);
            if ( v67 != v68 )
            {
              do
              {
                if ( v69 )
                  *v69 = *v67;
                ++v67;
                ++v69;
              }
              while ( v68 != v67 );
              LODWORD(v61) = v110.m128i_i32[2];
            }
            v110.m128i_i32[2] = v61 + v63;
            v70 = *(_QWORD *)(v51 + 40);
            for ( j = v70 + 40LL * *(unsigned int *)(v51 + 64); j != v70; v70 += 40 )
            {
              if ( *(_DWORD *)(*(_QWORD *)v70 + 24LL) == 9 )
              {
                v71 = *(_DWORD *)(*(_QWORD *)v70 + 96LL);
                if ( (unsigned int)(v71 - 1) <= 0x3FFFFFFE )
                {
                  v72 = v110.m128i_u32[2];
                  v37 = v110.m128i_u32[2] + 1LL;
                  if ( v37 > v110.m128i_u32[3] )
                  {
                    v89 = j;
                    v95 = *(_DWORD *)(*(_QWORD *)v70 + 96LL);
                    sub_C8D5F0((__int64)&v110, v111, v110.m128i_u32[2] + 1LL, 4u, j, v37);
                    v72 = v110.m128i_u32[2];
                    j = v89;
                    v71 = v95;
                  }
                  *(_DWORD *)(v110.m128i_i64[0] + 4 * v72) = v71;
                  ++v110.m128i_i32[2];
                }
              }
            }
          }
          while ( 1 )
          {
            v53 = *(_QWORD *)(v51 + 56);
            if ( !v53 )
              break;
            while ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v53 + 48LL) + 16LL * *(unsigned int *)(v53 + 8)) != 262 )
            {
              v53 = *(_QWORD *)(v53 + 32);
              if ( !v53 )
                goto LABEL_66;
            }
            v51 = *(_QWORD *)(v53 + 16);
            if ( !v51 )
              break;
            v52 = *(_DWORD *)(v51 + 24);
            if ( v52 != 50 )
              goto LABEL_94;
LABEL_114:
            v58 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v51 + 40) + 40LL) + 96LL);
            v59 = v110.m128i_u32[2];
            v60 = v110.m128i_u32[2] + 1LL;
            if ( v60 > v110.m128i_u32[3] )
            {
              sub_C8D5F0((__int64)&v110, v111, v60, 4u, j, v37);
              v59 = v110.m128i_u32[2];
            }
            *(_DWORD *)(v110.m128i_i64[0] + 4 * v59) = v58;
            ++v110.m128i_i32[2];
          }
        }
      }
    }
LABEL_66:
    if ( *((char *)v8 + 24) < 0 && (unsigned __int8)sub_B2D610(*(_QWORD *)*v6, 72) )
    {
      v73 = v6[4];
      v45 = v110.m128i_u32[2];
      v74 = *(__int64 (**)())(*(_QWORD *)v73 + 2392LL);
      if ( v74 != sub_302E270 )
      {
        v75 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v74)(v73, 72, v110.m128i_u32[2]);
        j = v75 + 2 * v76;
        v77 = (unsigned __int16 *)v75;
        v45 = v110.m128i_u32[2];
        if ( v75 != j )
        {
          v99 = v7;
          v78 = &v110;
          v79 = (unsigned __int16 *)j;
          do
          {
            v37 = v45 + 1;
            v80 = *v77;
            if ( v45 + 1 > (unsigned __int64)v110.m128i_u32[3] )
            {
              v96 = v78;
              sub_C8D5F0((__int64)v78, v111, v45 + 1, 4u, j, v37);
              v45 = v110.m128i_u32[2];
              v78 = v96;
            }
            ++v77;
            *(_DWORD *)(v110.m128i_i64[0] + 4 * v45) = v80;
            v45 = (unsigned int)++v110.m128i_i32[2];
          }
          while ( v79 != v77 );
          v7 = v99;
        }
      }
    }
    else
    {
      v45 = v110.m128i_u32[2];
    }
    if ( (_DWORD)v45 || *((_BYTE *)v8 + 9) || (v8[12] & 4) != 0 )
      sub_2E8FB70(v105, (unsigned int *)v110.m128i_i64[0], v45, v6[3]);
    if ( v100 == 32 && v101 )
    {
      v54 = v105;
      v106.m128i_i64[0] = v105;
      v106.m128i_i32[2] = sub_2E88FE0(v105) + *(unsigned __int8 *)(*(_QWORD *)(v54 + 16) + 9LL);
      v94 = v6;
      v55 = sub_2FC8970((__int64)&v106);
      v87 = v7;
      v56 = 0;
      do
      {
        if ( !*(_BYTE *)(*(_QWORD *)(v54 + 32) + 40LL * v55) )
          sub_2E89ED0(v54, v56++, v55);
        v55 = sub_2FC88B0(v54, v55);
      }
      while ( v101 > v56 );
      v6 = v94;
      v7 = v87;
    }
    v46 = *(_DWORD *)(v7 + 64);
    if ( v46 )
    {
      v47 = (unsigned int *)(*(_QWORD *)(v7 + 40) + 40LL * (unsigned int)(v46 - 1));
      v48 = *(_QWORD *)v47;
      if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v47 + 48LL) + 16LL * v47[2]) == 262 && *(_DWORD *)(v48 + 24) == -50 )
      {
        v49 = sub_3752000(v6, **(_QWORD **)(v48 + 40), *(_QWORD *)(*(_QWORD *)(v48 + 40) + 8LL), (__int64)a5, j, v37);
        v106.m128i_i64[0] = 0x20000000;
        v107 = 0;
        v106.m128i_i32[2] = v49;
        v108 = 0;
        v109 = 0;
        sub_2E8F270(v105, &v106);
      }
    }
    if ( (*((_BYTE *)v8 + 27) & 0x10) != 0 )
      (*(void (__fastcall **)(__int64, __int64, unsigned __int64))(*(_QWORD *)v6[4] + 2608LL))(v6[4], v105, v7);
    if ( (__int64 *)v110.m128i_i64[0] != v111 )
      _libc_free(v110.m128i_u64[0]);
  }
}
