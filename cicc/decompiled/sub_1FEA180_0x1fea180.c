// Function: sub_1FEA180
// Address: 0x1fea180
//
void __fastcall sub_1FEA180(size_t *a1, unsigned __int64 a2, unsigned __int8 a3, unsigned __int8 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rax
  unsigned __int64 v8; // r14
  int v9; // esi
  __int64 *v10; // r15
  __int64 v11; // r12
  unsigned int v12; // edx
  unsigned int *v13; // rax
  bool v14; // zf
  size_t v15; // rax
  __int64 v16; // rdi
  _QWORD *v17; // rax
  __int64 v18; // rdi
  _QWORD *v19; // rax
  __int64 v20; // rax
  bool v21; // cc
  _QWORD *v22; // rax
  int v23; // r8d
  int v24; // r9d
  unsigned int v25; // ecx
  __int64 v26; // rdx
  unsigned int v27; // ebx
  const __m128i *v28; // r12
  __int64 v29; // rax
  int v30; // r13d
  unsigned int v31; // r15d
  unsigned int v32; // eax
  __int32 v33; // r13d
  __int64 v34; // rax
  size_t v35; // rbx
  __int64 v36; // rsi
  __int64 *v37; // r13
  __int64 v38; // rcx
  __int64 *v39; // rdx
  __int64 v40; // r12
  int v41; // eax
  __int64 v42; // r15
  __int64 v43; // rdx
  __int64 v44; // rax
  unsigned int v45; // r12d
  __int64 v46; // rax
  int v47; // eax
  unsigned int v48; // ebx
  int v49; // r15d
  int v50; // r13d
  unsigned int v51; // edx
  unsigned int v52; // esi
  __int32 v53; // edx
  int *v54; // r12
  __int64 v55; // rbx
  int v56; // r14d
  __int64 v57; // r13
  unsigned int v58; // eax
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rbx
  __int64 *v62; // r12
  __int64 v63; // rdx
  __int64 v64; // rax
  _QWORD *v65; // rax
  unsigned __int64 v66; // rsi
  __int32 v67; // ebx
  __int32 v68; // r13d
  __int64 v69; // r12
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rsi
  __int64 *v73; // rbx
  __int64 v74; // r13
  __int64 v75; // r12
  __int64 v76; // rdx
  __int64 v77; // rax
  __int64 *v78; // [rsp+10h] [rbp-120h]
  const __m128i *v79; // [rsp+28h] [rbp-108h]
  unsigned int v82; // [rsp+34h] [rbp-FCh]
  unsigned int v83; // [rsp+38h] [rbp-F8h]
  __int64 v84; // [rsp+48h] [rbp-E8h]
  __int64 *v85; // [rsp+48h] [rbp-E8h]
  int v86; // [rsp+50h] [rbp-E0h]
  unsigned int v87; // [rsp+50h] [rbp-E0h]
  size_t v88; // [rsp+50h] [rbp-E0h]
  _QWORD *v89; // [rsp+50h] [rbp-E0h]
  __int64 v91; // [rsp+58h] [rbp-D8h]
  unsigned __int64 v92; // [rsp+58h] [rbp-D8h]
  __int64 v93; // [rsp+58h] [rbp-D8h]
  size_t v94; // [rsp+58h] [rbp-D8h]
  __int64 v95; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD *v96; // [rsp+68h] [rbp-C8h]
  __m128i v97; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v98; // [rsp+80h] [rbp-B0h]
  __int64 v99; // [rsp+88h] [rbp-A8h]
  __int64 v100; // [rsp+90h] [rbp-A0h]
  _BYTE *v101; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v102; // [rsp+A8h] [rbp-88h]
  _BYTE v103[32]; // [rsp+B0h] [rbp-80h] BYREF
  __m128i v104; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v105; // [rsp+E0h] [rbp-50h] BYREF
  _QWORD *v106; // [rsp+E8h] [rbp-48h]
  __int64 v107; // [rsp+F0h] [rbp-40h]

  v8 = a2;
  v9 = *(unsigned __int16 *)(a2 + 24);
  if ( (_WORD)v9 != 51 )
  {
    v10 = (__int64 *)a1;
    if ( (__int16)v9 <= 51 )
    {
      if ( (_WORD)v9 == 46 )
      {
        v65 = *(_QWORD **)(v8 + 32);
        v66 = v65[10];
        if ( *(_WORD *)(v66 + 24) == 8 )
        {
          v67 = *(_DWORD *)(v66 + 84);
        }
        else
        {
          v67 = sub_1FE6610(a1, v66, v65[11], a5);
          v65 = *(_QWORD **)(v8 + 32);
        }
        v68 = *(_DWORD *)(v65[5] + 84LL);
        if ( v67 != v68 )
        {
          v88 = a1[5];
          v85 = (__int64 *)a1[6];
          v93 = *(_QWORD *)(v88 + 56);
          v69 = (__int64)sub_1E0B640(v93, *(_QWORD *)(a1[2] + 8) + 960LL, (__int64 *)(v8 + 72), 0);
          sub_1DD5BA0((__int64 *)(v88 + 16), v69);
          v70 = *v85;
          v71 = *(_QWORD *)v69 & 7LL;
          *(_QWORD *)(v69 + 8) = v85;
          v70 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)v69 = v70 | v71;
          *(_QWORD *)(v70 + 8) = v69;
          *v85 = v69 | *v85 & 7;
          v104.m128i_i32[2] = v68;
          v104.m128i_i64[0] = 0x10000000;
          v105 = 0;
          v106 = 0;
          v107 = 0;
          sub_1E1A9C0(v69, v93, &v104);
          v104.m128i_i32[2] = v67;
          v104.m128i_i64[0] = 0;
          v105 = 0;
          v106 = 0;
          v107 = 0;
          sub_1E1A9C0(v69, v93, &v104);
        }
      }
      else if ( (_WORD)v9 == 47 )
      {
        sub_1FE9790((__int64 *)a1, v8, 0, a3, a4, *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL) + 84LL), a5);
      }
    }
    else if ( (__int16)v9 > 195 )
    {
      v14 = v9 == 239;
      v35 = a1[5];
      v36 = 1088;
      v37 = (__int64 *)a1[6];
      if ( !v14 )
        v36 = 1152;
      v38 = *(_QWORD *)(a1[2] + 8);
      v39 = (__int64 *)(v8 + 72);
      v40 = *(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL);
      v41 = *(unsigned __int16 *)(v40 + 24);
      if ( v41 != 14 && v41 != 36 )
      {
        v5 = (__int64)sub_1E0B640(*(_QWORD *)(v35 + 56), v38 + v36, v39, 0);
        sub_1DD5BA0((__int64 *)(v35 + 16), v5);
        v6 = *v37;
        v7 = *(_QWORD *)v5;
        *(_QWORD *)(v5 + 8) = v37;
        v6 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v5 = v6 | v7 & 7;
        *(_QWORD *)(v6 + 8) = v5;
        *v37 = *v37 & 7 | v5;
        BUG();
      }
      v91 = *(_QWORD *)(v35 + 56);
      v42 = (__int64)sub_1E0B640(v91, v38 + v36, v39, 0);
      sub_1DD5BA0((__int64 *)(v35 + 16), v42);
      v43 = *v37;
      v44 = *(_QWORD *)v42;
      *(_QWORD *)(v42 + 8) = v37;
      v43 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v42 = v43 | v44 & 7;
      *(_QWORD *)(v43 + 8) = v42;
      *v37 = v42 | *v37 & 7;
      LODWORD(v44) = *(_DWORD *)(v40 + 84);
      v104.m128i_i64[0] = 5;
      v105 = 0;
      LODWORD(v106) = v44;
      sub_1E1A9C0(v42, v91, &v104);
    }
    else if ( (__int16)v9 > 193 )
    {
      v14 = v9 == 194;
      v72 = 192;
      v73 = (__int64 *)a1[6];
      if ( !v14 )
        v72 = 320;
      v74 = *(_QWORD *)(a1[5] + 56);
      v89 = *(_QWORD **)(v8 + 88);
      v94 = a1[5];
      v75 = (__int64)sub_1E0B640(v74, *(_QWORD *)(a1[2] + 8) + v72, (__int64 *)(v8 + 72), 0);
      sub_1DD5BA0((__int64 *)(v94 + 16), v75);
      v76 = *v73;
      v77 = *(_QWORD *)v75;
      *(_QWORD *)(v75 + 8) = v73;
      v76 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v75 = v76 | v77 & 7;
      *(_QWORD *)(v76 + 8) = v75;
      *v73 = v75 | *v73 & 7;
      v104.m128i_i8[0] = 15;
      v105 = 0;
      v104.m128i_i32[0] &= 0xFFF000FF;
      v106 = v89;
      v104.m128i_i32[2] = 0;
      LODWORD(v107) = 0;
      sub_1E1A9C0(v75, v74, &v104);
    }
    else
    {
      v11 = *a1;
      v12 = *(_DWORD *)(v8 + 56) - 1;
      v13 = (unsigned int *)(*(_QWORD *)(v8 + 32) + 40LL * v12);
      v14 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v13 + 40LL) + 16LL * v13[2]) == 111;
      v15 = a1[2];
      v16 = *a1;
      if ( !v14 )
        v12 = *(_DWORD *)(v8 + 56);
      v82 = v12;
      v17 = sub_1E0B640(v16, *(_QWORD *)(v15 + 8) + 64LL, (__int64 *)(v8 + 72), 0);
      v95 = v11;
      v96 = v17;
      v18 = (__int64)v17;
      v19 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL) + 88LL);
      v104.m128i_i8[0] = 9;
      v105 = 0;
      v106 = v19;
      v104.m128i_i32[0] &= 0xFFF000FF;
      v104.m128i_i32[2] = 0;
      LODWORD(v107) = 0;
      sub_1E1A9C0(v18, v11, &v104);
      v20 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v8 + 32) + 120LL) + 88LL);
      v21 = *(_DWORD *)(v20 + 32) <= 0x40u;
      v22 = *(_QWORD **)(v20 + 24);
      if ( !v21 )
        v22 = (_QWORD *)*v22;
      v106 = v22;
      v104.m128i_i64[0] = 1;
      v105 = 0;
      sub_1E1A9C0((__int64)v96, v95, &v104);
      v101 = v103;
      v102 = 0x800000000LL;
      v104.m128i_i64[0] = (__int64)&v105;
      v104.m128i_i64[1] = 0x800000000LL;
      if ( v82 != 4 )
      {
        v78 = v10;
        v25 = 8;
        v26 = 0;
        v27 = 4;
        v28 = &v97;
        while ( 1 )
        {
          v29 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL * v27) + 88LL);
          if ( *(_DWORD *)(v29 + 32) <= 0x40u )
            v84 = *(_QWORD *)(v29 + 24);
          else
            v84 = **(_QWORD **)(v29 + 24);
          v86 = (unsigned __int16)v84 >> 3;
          v30 = *((_DWORD *)v96 + 10);
          if ( (unsigned int)v26 >= v25 )
          {
            sub_16CD150((__int64)&v101, v103, 0, 4, v23, v24);
            v26 = (unsigned int)v102;
          }
          v31 = v27 + 1;
          *(_DWORD *)&v101[4 * v26] = v30;
          v97.m128i_i64[0] = 1;
          v99 = (unsigned int)v84;
          LODWORD(v102) = v102 + 1;
          v98 = 0;
          sub_1E1A9C0((__int64)v96, v95, v28);
          v32 = v84 & 7;
          v83 = v32;
          if ( v32 != 2 )
            break;
          if ( !((unsigned __int16)v84 >> 3) )
            goto LABEL_63;
          v27 = v31 + v86;
          do
          {
            v53 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL * v31) + 84LL);
            v97.m128i_i64[0] = 0x10000000;
            v98 = 0;
            v97.m128i_i32[2] = v53;
            ++v31;
            v99 = 0;
            v100 = 0;
            *(__int32 *)((char *)v97.m128i_i32 + 3) = (unsigned __int8)(32 * (v53 > 0)) | 0x10;
            *(__int32 *)((char *)v97.m128i_i32 + 2) = v97.m128i_i16[1] & 0xF00F;
            v97.m128i_i32[0] &= 0xFFF000FF;
            sub_1E1A9C0((__int64)v96, v95, v28);
          }
          while ( v31 != v27 );
          if ( v82 == v27 )
          {
LABEL_50:
            v54 = (int *)v104.m128i_i64[0];
            v10 = v78;
            v55 = v104.m128i_i64[0] + 4LL * v104.m128i_u32[2];
            if ( v104.m128i_i64[0] != v55 )
            {
              v92 = v8;
              do
              {
                v56 = *v54;
                if ( (unsigned int)sub_1E165A0((__int64)v96, *v54, 0, v78[3]) != -1 )
                {
                  v57 = (__int64)v96;
                  v58 = sub_1E16810((__int64)v96, v56, 0, 0, v78[3]);
                  if ( v58 == -1 )
                  {
                    MEMORY[0] &= ~0x400000000uLL;
                    BUG();
                  }
                  v59 = *(_QWORD *)(v57 + 32) + 40LL * v58;
                  *(_BYTE *)(v59 + 4) &= ~4u;
                }
                ++v54;
              }
              while ( (int *)v55 != v54 );
              v8 = v92;
            }
            goto LABEL_57;
          }
LABEL_24:
          v26 = (unsigned int)v102;
          v25 = HIDWORD(v102);
        }
        if ( v32 <= 2 )
        {
          if ( !((unsigned __int16)v84 >> 3) )
          {
            ++v27;
LABEL_38:
            if ( (int)v84 < 0 )
            {
              v47 = *(_DWORD *)&v101[4 * (WORD1(v84) & 0x7FFF)];
              if ( (unsigned __int16)v84 >> 3 )
              {
                v87 = v27;
                v48 = v47 + 1;
                v49 = *(_DWORD *)&v101[4 * (unsigned int)v102 - 4] - v47;
                v50 = v47 + ((unsigned __int16)v84 >> 3) + 1;
                do
                {
                  v51 = v49 + v48;
                  v52 = v48++;
                  sub_1E16A40((__int64)v96, v52, v51);
                }
                while ( v50 != v48 );
                v27 = v87;
              }
            }
LABEL_23:
            if ( v82 == v27 )
              goto LABEL_50;
            goto LABEL_24;
          }
        }
        else
        {
          if ( v32 <= 4 )
          {
            if ( (unsigned __int16)v84 >> 3 )
            {
              v27 = v31 + v86;
              do
              {
                v33 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL * v31) + 84LL);
                v97.m128i_i64[0] = 0x410000000LL;
                v98 = 0;
                v97.m128i_i32[2] = v33;
                v99 = 0;
                v100 = 0;
                v97.m128i_i8[3] = (32 * (v33 > 0)) | 0x10;
                v97.m128i_i16[1] &= 0xF00Fu;
                v97.m128i_i32[0] &= 0xFFF000FF;
                sub_1E1A9C0((__int64)v96, v95, v28);
                v34 = v104.m128i_u32[2];
                if ( v104.m128i_i32[2] >= (unsigned __int32)v104.m128i_i32[3] )
                {
                  sub_16CD150((__int64)&v104, &v105, 0, 4, v23, v24);
                  v34 = v104.m128i_u32[2];
                }
                ++v31;
                *(_DWORD *)(v104.m128i_i64[0] + 4 * v34) = v33;
                ++v104.m128i_i32[2];
              }
              while ( v31 != v27 );
              goto LABEL_23;
            }
LABEL_63:
            ++v27;
            goto LABEL_23;
          }
          v24 = (unsigned __int16)v84 >> 3;
          if ( !((unsigned __int16)v84 >> 3) )
            goto LABEL_63;
        }
        v79 = v28;
        v45 = v27 + 1;
        v27 = v31 + v86;
        do
        {
          v46 = 5LL * v45++;
          sub_1FE6BA0(
            v78,
            &v95,
            *(_QWORD *)(*(_QWORD *)(v8 + 32) + 8 * v46),
            *(_QWORD *)(*(_QWORD *)(v8 + 32) + 8 * v46 + 8),
            0,
            0,
            a5,
            0,
            a3,
            a4);
        }
        while ( v45 != v27 );
        v28 = v79;
        if ( v83 != 1 )
          goto LABEL_23;
        goto LABEL_38;
      }
LABEL_57:
      v60 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v8 + 32) + 80LL) + 88LL);
      if ( v60 )
      {
        v97.m128i_i64[0] = 14;
        v98 = 0;
        v99 = v60;
        sub_1E1A9C0((__int64)v96, v95, &v97);
      }
      v61 = (__int64)v96;
      v62 = (__int64 *)v10[6];
      sub_1DD5BA0((__int64 *)(v10[5] + 16), (__int64)v96);
      v63 = *v62;
      v64 = *(_QWORD *)v61;
      *(_QWORD *)(v61 + 8) = v62;
      v63 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v61 = v63 | v64 & 7;
      *(_QWORD *)(v63 + 8) = v61;
      *v62 = *v62 & 7 | v61;
      if ( (__int64 *)v104.m128i_i64[0] != &v105 )
        _libc_free(v104.m128i_u64[0]);
      if ( v101 != v103 )
        _libc_free((unsigned __int64)v101);
    }
  }
}
