// Function: sub_20650A0
// Address: 0x20650a0
//
void __fastcall sub_20650A0(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _BYTE *v5; // r14
  __int64 v6; // rsi
  bool (__fastcall *v7)(__int64, __int64); // rax
  _DWORD *v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // eax
  int v11; // r8d
  int v12; // r9d
  signed __int64 v13; // rbx
  _BYTE *v14; // rdx
  _BYTE *v15; // rdi
  unsigned int v16; // ebx
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 *v20; // r15
  unsigned __int32 v21; // ecx
  _DWORD *v22; // r15
  __int64 v23; // rsi
  int v24; // eax
  _DWORD *v25; // rdx
  __int64 v26; // r14
  unsigned int v27; // ebx
  unsigned __int64 v28; // r15
  bool (__fastcall *v29)(__int64, __int64, __int64, unsigned __int64); // rax
  __int64 v30; // rdi
  __int64 v31; // rdx
  unsigned __int64 v32; // rax
  __int64 v33; // rdi
  int v34; // r8d
  int v35; // r9d
  __m128i *v36; // rdi
  signed __int64 v37; // rcx
  __int64 v38; // r13
  signed __int64 v39; // rbx
  __int64 v40; // rdx
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // r12
  bool (__fastcall *v43)(__int64, __int64, __int64, unsigned __int64); // rax
  __int64 v44; // rdi
  int v45; // eax
  int v46; // eax
  int v47; // r8d
  int v48; // r9d
  int v49; // r8d
  int v50; // r9d
  int v51; // ecx
  unsigned int v52; // edx
  unsigned int v53; // ecx
  unsigned int *v54; // rax
  const __m128i **v55; // rax
  unsigned int v56; // ebx
  signed __int64 v57; // r13
  unsigned int v58; // r12d
  const __m128i **v59; // r14
  __int64 v60; // rax
  __m128i *v61; // rax
  unsigned int v62; // r15d
  __int64 v63; // rax
  unsigned __int64 v64; // rax
  __int64 v65; // rax
  __m128i *v66; // rax
  unsigned __int64 v67; // rcx
  __int64 v68; // rax
  __int64 v69; // [rsp+28h] [rbp-188h]
  unsigned int v70; // [rsp+30h] [rbp-180h]
  __int64 v72; // [rsp+40h] [rbp-170h]
  signed __int64 v73; // [rsp+40h] [rbp-170h]
  __int64 n; // [rsp+48h] [rbp-168h]
  signed __int64 na; // [rsp+48h] [rbp-168h]
  signed __int64 v76; // [rsp+50h] [rbp-160h]
  __int64 v77; // [rsp+58h] [rbp-158h]
  __int64 v78; // [rsp+58h] [rbp-158h]
  _QWORD *v79; // [rsp+60h] [rbp-150h]
  unsigned int v81; // [rsp+78h] [rbp-138h]
  unsigned int v82; // [rsp+78h] [rbp-138h]
  unsigned __int32 v83; // [rsp+80h] [rbp-130h]
  __int64 v84; // [rsp+80h] [rbp-130h]
  __int64 v85; // [rsp+80h] [rbp-130h]
  __m128i v87; // [rsp+90h] [rbp-120h] BYREF
  __m128i v88; // [rsp+A0h] [rbp-110h] BYREF
  int v89; // [rsp+B0h] [rbp-100h]
  _BYTE *v90; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v91; // [rsp+C8h] [rbp-E8h]
  _BYTE v92[32]; // [rsp+D0h] [rbp-E0h] BYREF
  void *v93; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v94; // [rsp+F8h] [rbp-B8h]
  _BYTE v95[32]; // [rsp+100h] [rbp-B0h] BYREF
  void *v96; // [rsp+120h] [rbp-90h] BYREF
  __int64 v97; // [rsp+128h] [rbp-88h]
  _BYTE v98[32]; // [rsp+130h] [rbp-80h] BYREF
  __m128i v99; // [rsp+150h] [rbp-60h] BYREF
  __m128i v100; // [rsp+160h] [rbp-50h] BYREF
  int v101; // [rsp+170h] [rbp-40h]

  v5 = *(_BYTE **)(a1[69] + 16LL);
  v6 = *(_QWORD *)(*(_QWORD *)(a3 + 40) + 56LL);
  v7 = *(bool (__fastcall **)(__int64, __int64))(*(_QWORD *)v5 + 360LL);
  if ( v7 != sub_1F3D0C0 )
  {
    if ( !v7((__int64)v5, v6) )
      return;
LABEL_4:
    v77 = a2[1] - *a2;
    v10 = (*(__int64 (__fastcall **)(_BYTE *))(*(_QWORD *)v5 + 472LL))(v5);
    v70 = v10;
    if ( v77 <= 40 )
      return;
    v69 = v10;
    v13 = 0xCCCCCCCCCCCCCCCDLL * (v77 >> 3);
    v76 = v13;
    if ( v10 > v13 )
      return;
    v90 = v92;
    v91 = 0x800000000LL;
    n = 4LL * (unsigned int)v13;
    if ( v77 > 320 )
    {
      sub_16CD150((__int64)&v90, v92, v13, 4, v11, v12);
      v15 = v90;
      LODWORD(v91) = -858993459 * (v77 >> 3);
      v14 = &v90[n];
      if ( v90 == &v90[n] )
        goto LABEL_9;
    }
    else
    {
      LODWORD(v91) = -858993459 * (v77 >> 3);
      v14 = &v92[n];
      v15 = v92;
      if ( &v92[n] == v92 )
      {
LABEL_9:
        v72 = (__int64)v5;
        v16 = 0;
        v17 = 0;
        while ( 1 )
        {
          v18 = *a2 + 40 * v17;
          v19 = *(_QWORD *)(v18 + 16);
          v20 = (__int64 *)(*(_QWORD *)(v18 + 8) + 24LL);
          LODWORD(v97) = *(_DWORD *)(v19 + 32);
          if ( (unsigned int)v97 > 0x40 )
            sub_16A4FD0((__int64)&v96, (const void **)(v19 + 24));
          else
            v96 = *(void **)(v19 + 24);
          sub_16A7590((__int64)&v96, v20);
          v21 = v97;
          v22 = v96;
          LODWORD(v97) = 0;
          v23 = 4 * v17;
          v99.m128i_i32[2] = v21;
          v99.m128i_i64[0] = (__int64)v96;
          v83 = v21;
          if ( v21 <= 0x40 )
          {
            *(_DWORD *)&v90[4 * v17] = (_DWORD)v96 + 1;
            goto LABEL_11;
          }
          v24 = sub_16A57B0((__int64)&v99);
          v23 = 4 * v17;
          v25 = &v90[4 * v17];
          if ( v83 - v24 <= 0x40 )
          {
            *v25 = *v22 + 1;
          }
          else
          {
            *v25 = 0;
            if ( !v22 )
              goto LABEL_11;
          }
          j_j___libc_free_0_0(v22);
LABEL_11:
          if ( (unsigned int)v97 > 0x40 && v96 )
            j_j___libc_free_0_0(v96);
          if ( v16 )
            *(_DWORD *)&v90[v23] += *(_DWORD *)&v90[4 * v16 - 4];
          v17 = ++v16;
          if ( v16 >= v76 )
          {
            v26 = v72;
            v27 = v76 - 1;
            v28 = sub_20546A0((__int64)a1, a2, 0, (int)v76 - 1);
            v84 = sub_20547E0((__int64)a1, (__int64 *)&v90, 0, (int)v76 - 1);
            v29 = *(bool (__fastcall **)(__int64, __int64, __int64, unsigned __int64))(*(_QWORD *)v72 + 368LL);
            if ( v29 == sub_1F44290 )
            {
              v30 = *(_QWORD *)(*(_QWORD *)(a3 + 40) + 56LL) + 112LL;
              if ( (unsigned __int8)sub_1560180(v30, 34) || (unsigned __int8)sub_1560180(v30, 17) )
              {
                v31 = (unsigned int)sub_1F44230(v72, 1);
              }
              else
              {
                v82 = sub_1F44230(v72, 0);
                v46 = sub_1F44280(v72);
                v31 = v82;
                if ( v46 )
                {
                  LODWORD(v32) = sub_1F44280(v72);
                  v31 = v82;
                  v32 = (unsigned int)v32;
                  goto LABEL_38;
                }
              }
              v32 = 0xFFFFFFFFLL;
LABEL_38:
              if ( v28 > v32 || 100 * v84 < v31 * v28 )
              {
LABEL_40:
                v33 = a1[68];
                if ( (unsigned int)(*(_DWORD *)(v33 + 504) - 34) <= 1 )
                  goto LABEL_41;
LABEL_46:
                if ( !(unsigned int)sub_1700720(v33) )
                  goto LABEL_41;
                v93 = v95;
                v94 = 0x800000000LL;
                if ( v77 > 320 )
                {
                  sub_16CD150((__int64)&v93, v95, v76, 4, v34, v35);
                  LODWORD(v94) = v76;
                  if ( v93 != (char *)v93 + n )
                    memset(v93, 0, n);
                  v97 = 0x800000000LL;
                  v96 = v98;
                  sub_16CD150((__int64)&v96, v98, v76, 4, v47, v48);
                  LODWORD(v97) = v76;
                  if ( v96 != (char *)v96 + n )
                    memset(v96, 0, n);
                  v99.m128i_i64[0] = (__int64)&v100;
                  v99.m128i_i64[1] = 0x800000000LL;
                  sub_16CD150((__int64)&v99, &v100, v76, 4, v49, v50);
                  v36 = (__m128i *)v99.m128i_i64[0];
                }
                else
                {
                  LODWORD(v94) = v76;
                  if ( v95 != &v95[n] )
                    memset(v95, 0, n);
                  v96 = v98;
                  v97 = (unsigned int)v76 | 0x800000000LL;
                  if ( v98 != &v98[n] )
                    memset(v98, 0, n);
                  v99.m128i_i32[3] = 8;
                  v99.m128i_i64[0] = (__int64)&v100;
                  v36 = &v100;
                }
                v99.m128i_i32[2] = v76;
                if ( n )
                  memset(v36, 0, n);
                v37 = v76 - 1;
                *((_DWORD *)v93 + v37) = 1;
                na = v76 - 1;
                *((_DWORD *)v96 + v37) = v27;
                *(_DWORD *)(v99.m128i_i64[0] + 4 * v37) = 2;
                v73 = v76 - 2;
                if ( v76 == 1 )
                  goto LABEL_89;
                v79 = a2;
                while ( 1 )
                {
                  v78 = 4 * v73;
                  v38 = v76 - v73;
                  *((_DWORD *)v93 + v73) = *((_DWORD *)v93 + v73 + 1) + 1;
                  *((_DWORD *)v96 + v73) = v73;
                  *(_DWORD *)(v99.m128i_i64[0] + 4 * v73) = *(_DWORD *)(v99.m128i_i64[0] + 4 * v73 + 4) + 2;
                  v39 = v76 - 1;
                  if ( na > v73 )
                    break;
LABEL_87:
                  if ( --v73 == -1 )
                  {
                    a2 = v79;
LABEL_89:
                    v55 = (const __m128i **)a2;
                    v56 = 0;
                    v57 = 0;
                    v58 = 0;
                    v59 = v55;
                    do
                    {
                      v62 = *((_DWORD *)v96 + v57);
                      v89 = -1;
                      if ( v70 <= v62 + 1 - v56
                        && (unsigned __int8)sub_2063B90(a1, v59, v56, v62, a3, a4, (__int64)&v87) )
                      {
                        v60 = v58++;
                        v61 = (__m128i *)((char *)*v59 + 40 * v60);
                        *v61 = _mm_loadu_si128(&v87);
                        v61[1] = _mm_loadu_si128(&v88);
                        v61[2].m128i_i32[0] = v89;
                      }
                      else if ( v56 <= v62 )
                      {
                        while ( 1 )
                        {
                          ++v56;
                          v63 = 5LL * v58++;
                          memmove((char *)*v59 + 8 * v63, (char *)*v59 + 40 * v57, 0x28u);
                          if ( v62 < v56 )
                            break;
                          v57 = v56;
                        }
                      }
                      v57 = v62 + 1;
                      v56 = v62 + 1;
                    }
                    while ( v57 < v76 );
                    v64 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v59[1] - (char *)*v59) >> 3);
                    if ( v64 < v58 )
                    {
                      sub_205A2F0(v59, v58 - v64);
                    }
                    else if ( v64 > v58 )
                    {
                      v65 = (__int64)&(*v59)->m128i_i64[5 * v58];
                      if ( v59[1] != (const __m128i *)v65 )
                        v59[1] = (const __m128i *)v65;
                    }
                    if ( (__m128i *)v99.m128i_i64[0] != &v100 )
                      _libc_free(v99.m128i_u64[0]);
                    if ( v96 != v98 )
                      _libc_free((unsigned __int64)v96);
                    if ( v93 != v95 )
                      _libc_free((unsigned __int64)v93);
LABEL_41:
                    if ( v90 != v92 )
                      _libc_free((unsigned __int64)v90);
                    return;
                  }
                }
                while ( 1 )
                {
                  v42 = sub_20546A0((__int64)a1, v79, v73, v39);
                  v85 = sub_20547E0((__int64)a1, (__int64 *)&v90, v73, v39);
                  v43 = *(bool (__fastcall **)(__int64, __int64, __int64, unsigned __int64))(*(_QWORD *)v26 + 368LL);
                  if ( v43 != sub_1F44290 )
                  {
                    if ( !v43(v26, a3, v85, v42) )
                      goto LABEL_63;
                    goto LABEL_78;
                  }
                  v44 = *(_QWORD *)(*(_QWORD *)(a3 + 40) + 56LL) + 112LL;
                  if ( (unsigned __int8)sub_1560180(v44, 34) || (unsigned __int8)sub_1560180(v44, 17) )
                  {
                    v40 = (unsigned int)sub_1F44230(v26, 1);
                  }
                  else
                  {
                    v81 = sub_1F44230(v26, 0);
                    v45 = sub_1F44280(v26);
                    v40 = v81;
                    if ( v45 )
                    {
                      LODWORD(v41) = sub_1F44280(v26);
                      v40 = v81;
                      v41 = (unsigned int)v41;
                      goto LABEL_61;
                    }
                  }
                  v41 = 0xFFFFFFFFLL;
LABEL_61:
                  if ( v42 > v41 || 100 * v85 < v40 * v42 )
                    goto LABEL_63;
LABEL_78:
                  if ( na == v39 )
                  {
                    v52 = 1;
                    v51 = 0;
                  }
                  else
                  {
                    v51 = *(_DWORD *)(v99.m128i_i64[0] + 4 * (v39 + 1));
                    v52 = *((_DWORD *)v93 + v39 + 1) + 1;
                  }
                  if ( v38 == 1 )
                  {
                    v53 = v51 + 2;
                  }
                  else if ( v70 >> 1 < v38 )
                  {
                    v53 = (v69 <= v38) + v51;
                  }
                  else
                  {
                    v53 = v51 + 1;
                  }
                  v54 = (unsigned int *)((char *)v93 + v78);
                  if ( *(_DWORD *)((char *)v93 + v78) <= v52
                    && (*v54 != v52 || *(_DWORD *)(v99.m128i_i64[0] + 4 * v73) >= v53) )
                  {
LABEL_63:
                    --v38;
                    --v39;
                    if ( v38 == 1 )
                      goto LABEL_87;
                    continue;
                  }
                  *v54 = v52;
                  --v38;
                  *((_DWORD *)v96 + v73) = v39--;
                  *(_DWORD *)(v99.m128i_i64[0] + 4 * v73) = v53;
                  if ( v38 == 1 )
                    goto LABEL_87;
                }
              }
            }
            else if ( !v29(v72, a3, v84, v28) )
            {
              goto LABEL_40;
            }
            v101 = -1;
            if ( !(unsigned __int8)sub_2063B90(a1, a2, 0, v27, a3, a4, (__int64)&v99) )
            {
              v33 = a1[68];
              goto LABEL_46;
            }
            v66 = (__m128i *)*a2;
            *v66 = _mm_loadu_si128(&v99);
            v66[1] = _mm_loadu_si128(&v100);
            v66[2].m128i_i32[0] = v101;
            v67 = a2[1] - *a2;
            if ( v67 )
            {
              if ( v67 > 0x28 )
              {
                v68 = *a2 + 40LL;
                if ( a2[1] != v68 )
                  a2[1] = v68;
              }
            }
            else
            {
              sub_205A2F0((const __m128i **)a2, 1u);
            }
            goto LABEL_41;
          }
        }
      }
    }
    memset(v15, 0, v14 - v15);
    goto LABEL_9;
  }
  v99.m128i_i64[0] = sub_1560340((_QWORD *)(v6 + 112), -1, "no-jump-tables", 0xEu);
  v8 = (_DWORD *)sub_155D8B0(v99.m128i_i64);
  if ( (v9 != 4 || *v8 != 1702195828) && ((v5[2871] & 0xFB) == 0 || (v5[2870] & 0xFB) == 0) )
    goto LABEL_4;
}
