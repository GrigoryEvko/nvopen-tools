// Function: sub_2050700
// Address: 0x2050700
//
__int64 __fastcall sub_2050700(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        unsigned int a6,
        __int64 *a7)
{
  char v8; // r12
  __int64 v10; // r13
  __int64 result; // rax
  int v13; // edx
  __int64 v14; // r13
  __int64 (*v15)(void); // rax
  __int64 v16; // rdi
  int v17; // eax
  char v18; // cl
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int16 v21; // ax
  bool i; // cc
  int v23; // r15d
  __int8 v24; // al
  char v25; // r15
  __int64 v26; // rbx
  __int64 v27; // r12
  int v28; // r8d
  int v29; // r9d
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r12
  __int64 v33; // rdx
  int v34; // eax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rsi
  unsigned int v38; // edx
  __int64 *v39; // r10
  __int64 v40; // rdi
  __int64 v41; // r10
  __int64 v42; // rax
  _DWORD *v43; // rcx
  _DWORD *v44; // rax
  int v45; // edx
  __int32 v46; // eax
  unsigned int v47; // ecx
  unsigned int v48; // eax
  __int64 v49; // r8
  __int64 v50; // r15
  __int64 *v51; // r15
  __int64 v52; // rcx
  int v53; // edx
  __int32 *v54; // r12
  unsigned int v55; // r15d
  __int64 v56; // rdx
  int v57; // r9d
  unsigned int *v58; // r10
  __int64 v59; // r8
  __int64 v60; // rax
  __int64 v61; // rsi
  __int32 v62; // eax
  unsigned int v63; // ebx
  __int64 v64; // r11
  unsigned __int64 v65; // rsi
  __int64 v66; // r10
  unsigned int v67; // r9d
  int v68; // eax
  int v69; // r10d
  int v70; // r9d
  unsigned __int64 v71; // rax
  __int64 v72; // rcx
  __int64 v73; // rax
  __int64 v74; // rcx
  unsigned int v75; // r11d
  __int64 v76; // r10
  unsigned int v77; // r9d
  __int64 v78; // r8
  __int64 v79; // rdx
  __int64 v80; // [rsp+0h] [rbp-1C0h]
  __int64 v81; // [rsp+10h] [rbp-1B0h]
  unsigned int *v82; // [rsp+10h] [rbp-1B0h]
  __int64 v83; // [rsp+10h] [rbp-1B0h]
  __int64 v84; // [rsp+18h] [rbp-1A8h]
  __int64 v85; // [rsp+18h] [rbp-1A8h]
  __int64 v86; // [rsp+20h] [rbp-1A0h]
  __int32 v87; // [rsp+20h] [rbp-1A0h]
  __int64 v88; // [rsp+20h] [rbp-1A0h]
  unsigned int v89; // [rsp+20h] [rbp-1A0h]
  int v90; // [rsp+28h] [rbp-198h]
  __int64 v91; // [rsp+28h] [rbp-198h]
  __int64 v92; // [rsp+28h] [rbp-198h]
  __int64 v93; // [rsp+28h] [rbp-198h]
  __int64 v94; // [rsp+28h] [rbp-198h]
  __int64 v95; // [rsp+30h] [rbp-190h]
  __int32 *v96; // [rsp+30h] [rbp-190h]
  unsigned int v97; // [rsp+30h] [rbp-190h]
  __int64 v98; // [rsp+30h] [rbp-190h]
  unsigned int v99; // [rsp+30h] [rbp-190h]
  __int64 v100; // [rsp+38h] [rbp-188h]
  __int64 v101; // [rsp+38h] [rbp-188h]
  __int64 v102; // [rsp+38h] [rbp-188h]
  __int64 v103; // [rsp+38h] [rbp-188h]
  __int64 v104; // [rsp+38h] [rbp-188h]
  char v108; // [rsp+5Fh] [rbp-161h]
  __int64 v109; // [rsp+68h] [rbp-158h] BYREF
  __int64 v110; // [rsp+70h] [rbp-150h] BYREF
  char v111; // [rsp+78h] [rbp-148h]
  __m128i v112; // [rsp+80h] [rbp-140h] BYREF
  __int64 v113; // [rsp+90h] [rbp-130h]
  __int64 v114; // [rsp+98h] [rbp-128h]
  __int64 v115; // [rsp+A0h] [rbp-120h]
  unsigned __int8 v116; // [rsp+A8h] [rbp-118h]
  __int32 *v117; // [rsp+B0h] [rbp-110h] BYREF
  int v118; // [rsp+B8h] [rbp-108h]
  char v119; // [rsp+C0h] [rbp-100h] BYREF
  __int64 v120[2]; // [rsp+E0h] [rbp-E0h] BYREF
  _BYTE v121[64]; // [rsp+F0h] [rbp-D0h] BYREF
  char *v122; // [rsp+130h] [rbp-90h]
  char v123; // [rsp+140h] [rbp-80h] BYREF
  _BYTE *v124; // [rsp+148h] [rbp-78h]
  _BYTE v125[16]; // [rsp+158h] [rbp-68h] BYREF
  _BYTE *v126; // [rsp+168h] [rbp-58h]
  int v127; // [rsp+170h] [rbp-50h]
  _BYTE v128[72]; // [rsp+178h] [rbp-48h] BYREF

  v8 = a6;
  v108 = a6;
  if ( !(_BYTE)a6 )
  {
    v10 = *(_QWORD *)(a1 + 712);
    result = 0;
    if ( *(_QWORD *)(v10 + 784) != *(_QWORD *)(*(_QWORD *)(v10 + 8) + 328LL) )
      return result;
    v13 = *(_DWORD *)(a1 + 536);
    if ( !*(_WORD *)(a3 + 32) || *(_DWORD *)(a5 + 8) == 2 && *(_QWORD *)(a5 - 8) )
    {
      if ( v13 != 1 )
        return result;
    }
    else
    {
      v47 = *(_DWORD *)(a2 + 32);
      v48 = *(_DWORD *)(v10 + 496);
      v49 = 1LL << v47;
      v50 = 8LL * (v47 >> 6);
      if ( v47 >= v48 )
      {
        v64 = *(_QWORD *)(v10 + 488);
        v65 = v47 + 1;
        v66 = v10 + 480;
        v67 = v47 + 1;
        if ( v65 > v64 << 6 )
        {
          v83 = 1LL << v47;
          v71 = (v47 + 64) >> 6;
          v72 = 2 * v64;
          v93 = *(_QWORD *)(v10 + 488);
          if ( v71 >= 2 * v64 )
            v72 = v71;
          v98 = v72;
          v103 = 8 * v72;
          v73 = (__int64)realloc(*(_QWORD *)(v10 + 480), 8 * v72, 8 * (int)v72, v72, v49, v65);
          v74 = v98;
          v75 = v93;
          v76 = v10 + 480;
          v77 = v65;
          v78 = v83;
          if ( !v73 )
          {
            if ( v103 )
            {
              sub_16BD1C0("Allocation failed", 1u);
              v74 = v98;
              v73 = 0;
              v75 = v93;
              v78 = v83;
              v77 = v65;
              v76 = v10 + 480;
            }
            else
            {
              v73 = malloc(1u);
              v76 = v10 + 480;
              v77 = v65;
              v78 = v83;
              v75 = v93;
              v74 = v98;
              if ( !v73 )
              {
                sub_16BD1C0("Allocation failed", 1u);
                v76 = v10 + 480;
                v77 = v65;
                v78 = v83;
                v75 = v93;
                v73 = 0;
                v74 = v98;
              }
            }
          }
          *(_QWORD *)(v10 + 480) = v73;
          *(_QWORD *)(v10 + 488) = v74;
          v89 = v75;
          v94 = v78;
          v99 = v77;
          v104 = v76;
          sub_13A4C60(v76, 0);
          v66 = v104;
          v67 = v99;
          v49 = v94;
          v79 = *(_QWORD *)(v10 + 488) - v89;
          if ( v79 )
          {
            memset((void *)(*(_QWORD *)(v10 + 480) + 8LL * v89), 0, 8 * v79);
            v49 = v94;
            v67 = v99;
            v66 = v104;
          }
          v48 = *(_DWORD *)(v10 + 496);
        }
        if ( v67 > v48 )
        {
          v92 = v49;
          v97 = v67;
          v102 = v66;
          sub_13A4C60(v66, 0);
          v48 = *(_DWORD *)(v10 + 496);
          v49 = v92;
          v67 = v97;
          v66 = v102;
        }
        *(_DWORD *)(v10 + 496) = v67;
        if ( v67 < v48 )
        {
          v101 = v49;
          sub_13A4C60(v66, 0);
          v49 = v101;
        }
        v51 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 712) + 480LL) + v50);
        v52 = *v51;
      }
      else
      {
        v51 = (__int64 *)(*(_QWORD *)(v10 + 480) + v50);
        v52 = *v51;
        if ( v13 != 1 )
        {
          result = a6;
          if ( (v52 & v49) != 0 )
            return result;
        }
      }
      *v51 = v49 | v52;
    }
  }
  v100 = 0;
  v14 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL);
  v15 = *(__int64 (**)(void))(**(_QWORD **)(v14 + 16) + 40LL);
  if ( v15 != sub_1D00B00 )
    v100 = v15();
  v16 = *(_QWORD *)(a1 + 712);
  v116 = 0;
  v17 = sub_1FDEA40(v16, a2);
  if ( v17 != 0x7FFFFFFF )
  {
    LODWORD(v114) = v17;
    v113 = 0;
    v112.m128i_i32[0] = v112.m128i_i32[0] & 0xFFF00000 | 5;
    v116 = 1;
    v108 = 0;
    v24 = v112.m128i_i8[0];
    goto LABEL_19;
  }
  v18 = v116;
  v19 = *a7;
  if ( *a7 )
  {
    v20 = *a7;
    v21 = *(_WORD *)(v19 + 24);
    for ( i = v21 <= 47; v21 != 47; i = v21 <= 47 )
    {
      if ( i )
      {
        if ( (unsigned __int16)(v21 - 3) > 1u )
          goto LABEL_27;
      }
      else if ( v21 != 145 && v21 != 158 )
      {
        goto LABEL_27;
      }
      v20 = **(_QWORD **)(v20 + 32);
      v21 = *(_WORD *)(v20 + 24);
    }
    v23 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v20 + 32) + 40LL) + 84LL);
    if ( v23 )
    {
      if ( v23 < 0 )
      {
        v68 = sub_1E69F00(*(_QWORD *)(v14 + 40), v23);
        if ( v68 )
          v23 = v68;
      }
      v112.m128i_i64[0] &= 0xFFFFFFF000000000LL;
      v112.m128i_i32[2] = v23;
      v113 = 0;
      v114 = 0;
      v115 = 0;
      v116 = 1;
      v24 = v112.m128i_i8[0];
      goto LABEL_19;
    }
LABEL_27:
    if ( *(_WORD *)(v19 + 24) == 185 )
    {
      v33 = *(_QWORD *)(*(_QWORD *)(v19 + 32) + 40LL);
      v34 = *(unsigned __int16 *)(v33 + 24);
      if ( v34 == 14 || v34 == 36 )
      {
        v53 = *(_DWORD *)(v33 + 84);
        v113 = 0;
        v25 = 1;
        v116 = 1;
        LODWORD(v114) = v53;
        v112.m128i_i32[0] = v112.m128i_i32[0] & 0xFFF00000 | 5;
LABEL_21:
        v26 = *(_QWORD *)(a1 + 712);
        v27 = *(_QWORD *)(v100 + 8);
        sub_15C7080(v120, a5);
        sub_1E1C280(v14, v120, v27 + 768, v25, &v112, a3, (__int64)a4);
        v30 = *(unsigned int *)(v26 + 408);
        v32 = v31;
        if ( (unsigned int)v30 >= *(_DWORD *)(v26 + 412) )
        {
          sub_16CD150(v26 + 400, (const void *)(v26 + 416), 0, 8, v28, v29);
          v30 = *(unsigned int *)(v26 + 408);
        }
        *(_QWORD *)(*(_QWORD *)(v26 + 400) + 8 * v30) = v32;
        ++*(_DWORD *)(v26 + 408);
        if ( v120[0] )
          sub_161E7C0((__int64)v120, v120[0]);
        return 1;
      }
    }
  }
  v35 = *(_QWORD *)(a1 + 712);
  v36 = *(unsigned int *)(v35 + 232);
  if ( (_DWORD)v36 )
  {
    v37 = *(_QWORD *)(v35 + 216);
    v38 = (v36 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v39 = (__int64 *)(v37 + 16LL * v38);
    v40 = *v39;
    if ( a2 == *v39 )
    {
LABEL_32:
      if ( v39 != (__int64 *)(16 * v36 + v37) )
      {
        v81 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL);
        sub_2043DE0((__int64)&v117, a2);
        v86 = v41;
        v90 = *(_DWORD *)(v41 + 8);
        v84 = *(_QWORD *)a2;
        v95 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
        v42 = sub_16498A0(a2);
        sub_204E3C0((__int64)v120, v42, v81, v95, v90, v84, (unsigned int *)&v117);
        v43 = &v126[4 * v127];
        if ( v126 != (_BYTE *)v43 )
        {
          v44 = v126;
          v45 = 0;
          do
            v45 += *v44++;
          while ( v43 != v44 );
          if ( v45 > 1 )
          {
            sub_20505F0((__int64)&v117, (__int64)v120);
            v54 = v117;
            v96 = &v117[2 * v118];
            if ( v117 != v96 )
            {
              v55 = 0;
              v80 = a1;
              do
              {
                v62 = *v54;
                v63 = v54[1];
                v112.m128i_i64[0] &= 0xFFFFFFF000000000LL;
                v112.m128i_i32[2] = v62;
                v113 = 0;
                v114 = 0;
                v115 = 0;
                if ( !v116 )
                  v116 = 1;
                sub_15C4EF0((__int64)&v110, a4, v55, v63);
                if ( v111 )
                {
                  v87 = v112.m128i_i32[2];
                  v85 = v110;
                  v82 = *(unsigned int **)(v80 + 712);
                  v91 = *(_QWORD *)(v100 + 8) + 768LL;
                  sub_15C7080(&v109, a5);
                  sub_1E1C0A0(v14, &v109, v91, v108, v87, a3, v85);
                  v58 = v82;
                  v59 = v56;
                  v60 = v82[102];
                  if ( (unsigned int)v60 >= v82[103] )
                  {
                    v88 = v56;
                    sub_16CD150((__int64)(v82 + 100), v82 + 104, 0, 8, v56, v57);
                    v58 = v82;
                    v59 = v88;
                    v60 = v82[102];
                  }
                  *(_QWORD *)(*((_QWORD *)v58 + 50) + 8 * v60) = v59;
                  v61 = v109;
                  ++v58[102];
                  if ( v61 )
                    sub_161E7C0((__int64)&v109, v61);
                  v55 += v63;
                }
                v54 += 2;
              }
              while ( v96 != v54 );
              v96 = v117;
            }
            if ( v96 != (__int32 *)&v119 )
              _libc_free((unsigned __int64)v96);
            if ( v126 != v128 )
              _libc_free((unsigned __int64)v126);
            if ( v124 != v125 )
              _libc_free((unsigned __int64)v124);
            if ( v122 != &v123 )
              _libc_free((unsigned __int64)v122);
            if ( (_BYTE *)v120[0] != v121 )
              _libc_free(v120[0]);
            return 1;
          }
        }
        v46 = *(_DWORD *)(v86 + 8);
        v112.m128i_i64[0] &= 0xFFFFFFF000000000LL;
        v112.m128i_i32[2] = v46;
        v113 = 0;
        v114 = 0;
        v115 = 0;
        v116 = 1;
        if ( v126 != v128 )
          _libc_free((unsigned __int64)v126);
        if ( v124 != v125 )
          _libc_free((unsigned __int64)v124);
        if ( v122 != &v123 )
          _libc_free((unsigned __int64)v122);
        if ( (_BYTE *)v120[0] != v121 )
          _libc_free(v120[0]);
        v18 = v8;
      }
    }
    else
    {
      v69 = 1;
      while ( v40 != -8 )
      {
        v70 = v69 + 1;
        v38 = (v36 - 1) & (v69 + v38);
        v39 = (__int64 *)(v37 + 16LL * v38);
        v40 = *v39;
        if ( a2 == *v39 )
          goto LABEL_32;
        v69 = v70;
      }
    }
  }
  result = v116;
  if ( v116 )
  {
    v108 = v18;
    v24 = v112.m128i_i8[0];
LABEL_19:
    v25 = v108;
    if ( v24 )
      v25 = 1;
    goto LABEL_21;
  }
  return result;
}
