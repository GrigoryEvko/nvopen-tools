// Function: sub_2F6B510
// Address: 0x2f6b510
//
__int64 __fastcall sub_2F6B510(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  _DWORD *v8; // rax
  unsigned __int64 v9; // r12
  __int64 v10; // r14
  __int64 v13; // rdi
  unsigned int v14; // esi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned int v19; // esi
  __int64 *v20; // rdx
  __int64 *v21; // rbx
  __int64 v22; // rax
  __int64 *v23; // r14
  signed __int64 v24; // rbx
  __int64 *v25; // rdx
  __int64 v26; // rcx
  __int64 *v27; // rdi
  unsigned int *v28; // r8
  __int64 v29; // r9
  unsigned __int64 v30; // rsi
  __int64 v31; // rdi
  __int64 v32; // r15
  __int64 v33; // rdi
  unsigned int v34; // edi
  __int64 v35; // r10
  unsigned __int64 v36; // rax
  __int64 v37; // r10
  __int64 *v38; // r15
  __int64 v39; // rbx
  __int64 v40; // rax
  bool v41; // al
  unsigned __int8 *v42; // rsi
  __int32 v43; // r8d
  __int64 v44; // rax
  __int64 v45; // rbx
  _QWORD *v46; // rsi
  __int64 v47; // rdx
  __int32 v48; // eax
  unsigned __int64 v49; // rbx
  __int64 v50; // rcx
  __int64 v51; // r8
  unsigned __int64 v52; // r9
  __int64 v53; // rcx
  __int64 v54; // r12
  __int64 v55; // rbx
  __int64 *v56; // rsi
  __int64 **v57; // rax
  char v58; // bl
  __int64 v59; // r12
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  _QWORD *v63; // r14
  __int64 v64; // r12
  unsigned int v65; // edx
  __int64 v66; // rcx
  __int64 v67; // rdi
  unsigned int v68; // eax
  __int64 *v69; // rcx
  _QWORD *v70; // r8
  __int64 v71; // rax
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 v75; // rbx
  __int64 v76; // rax
  __int64 v77; // rdi
  __int64 v78; // rdi
  __int64 *v79; // rax
  unsigned __int8 v80; // [rsp+Fh] [rbp-121h]
  unsigned __int64 v81; // [rsp+10h] [rbp-120h]
  unsigned __int8 v82; // [rsp+18h] [rbp-118h]
  __int32 v83; // [rsp+18h] [rbp-118h]
  unsigned __int64 v84; // [rsp+20h] [rbp-110h]
  __int64 v85; // [rsp+28h] [rbp-108h]
  __int64 v86; // [rsp+30h] [rbp-100h]
  __int64 i; // [rsp+38h] [rbp-F8h]
  __int64 v88; // [rsp+38h] [rbp-F8h]
  unsigned __int64 v89; // [rsp+38h] [rbp-F8h]
  __int64 *v90; // [rsp+40h] [rbp-F0h]
  __int64 v91; // [rsp+40h] [rbp-F0h]
  __int64 v92; // [rsp+48h] [rbp-E8h]
  unsigned __int8 *v93; // [rsp+58h] [rbp-D8h] BYREF
  __int64 *v94; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v95; // [rsp+68h] [rbp-C8h]
  _QWORD v96[8]; // [rsp+70h] [rbp-C0h] BYREF
  __m128i v97; // [rsp+B0h] [rbp-80h] BYREF
  _QWORD v98[14]; // [rsp+C0h] [rbp-70h] BYREF

  if ( *(_WORD *)(a3 + 68) != 20 )
    return 0;
  v8 = *(_DWORD **)(a3 + 32);
  v9 = a3;
  if ( (*v8 & 0xFFF00) != 0 )
    return 0;
  if ( (v8[10] & 0xFFF00) != 0 )
    return 0;
  v10 = *(_QWORD *)(a3 + 24);
  if ( *(_BYTE *)(v10 + 216) || *(_BYTE *)(v10 + 262) || *(_DWORD *)(v10 + 72) != 2 )
    return 0;
  v13 = *(_QWORD *)(a1 + 40);
  if ( *(_BYTE *)(a2 + 26) )
    v14 = *(_DWORD *)(a2 + 8);
  else
    v14 = *(_DWORD *)(a2 + 12);
  v85 = sub_2DF8570(v13, v14, a3, a4, a5, a6);
  if ( *(_BYTE *)(a2 + 26) )
    v19 = *(_DWORD *)(a2 + 12);
  else
    v19 = *(_DWORD *)(a2 + 8);
  v92 = sub_2DF8570(*(_QWORD *)(a1 + 40), v19, v15, v16, v17, v18);
  v84 = sub_2DF8360(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), v9, 0) & 0xFFFFFFFFFFFFFFF8LL;
  v86 = v84 | 2;
  v20 = (__int64 *)sub_2E09D00((__int64 *)v85, v84 | 2);
  if ( v20 == (__int64 *)(*(_QWORD *)v85 + 24LL * *(unsigned int *)(v85 + 8))
    || (*(_DWORD *)((*v20 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v20 >> 1) & 3) > (*(_DWORD *)(v84 + 24) | 1u) )
  {
    BUG();
  }
  if ( (*(_BYTE *)(v20[2] + 8) & 6) != 0 )
    return 0;
  if ( sub_2E0A0C0(
         v92,
         *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 152LL) + 16LL * *(unsigned int *)(v10 + 24)),
         v86) )
  {
    return 0;
  }
  v21 = *(__int64 **)(v10 + 64);
  v90 = &v21[*(unsigned int *)(v10 + 72)];
  if ( v90 == v21 )
    return 0;
  v81 = v9;
  v22 = *(_QWORD *)(a1 + 40);
  v82 = 0;
  v23 = *(__int64 **)(v10 + 64);
  for ( i = 0; ; i = v32 )
  {
    v32 = *v23;
    v33 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v22 + 32) + 152LL) + 16LL * *(unsigned int *)(*v23 + 24) + 8);
    if ( ((v33 >> 1) & 3) != 0 )
      v24 = v33 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v33 >> 1) & 3) - 1));
    else
      v24 = *(_QWORD *)(v33 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
    v27 = (__int64 *)sub_2E09D00((__int64 *)v85, v24);
    if ( v27 == (__int64 *)(*(_QWORD *)v85 + 24LL * *(unsigned int *)(v85 + 8))
      || (v28 = (unsigned int *)(v24 & 0xFFFFFFFFFFFFFFF8LL),
          (*(_DWORD *)((*v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v27 >> 1) & 3) > (*(_DWORD *)((v24 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)((v24 >> 1) & 3))) )
    {
      BUG();
    }
    v22 = *(_QWORD *)(a1 + 40);
    v29 = *(_QWORD *)(v27[2] + 8);
    v30 = v29 & 0xFFFFFFFFFFFFFFF8LL;
    v31 = *(_QWORD *)((v29 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    if ( v31 )
    {
      if ( *(_WORD *)(v31 + 68) == 20 )
      {
        v28 = *(unsigned int **)(v31 + 32);
        if ( (*v28 & 0xFFF00) == 0 && (v28[10] & 0xFFF00) == 0 )
        {
          v26 = v28[2];
          if ( *(_DWORD *)(v85 + 112) == (_DWORD)v26 )
          {
            v26 = *(unsigned int *)(v92 + 112);
            if ( v28[12] == (_DWORD)v26 && v32 == *(_QWORD *)(v31 + 24) )
            {
              v28 = *(unsigned int **)(v92 + 64);
              v26 = (__int64)&v28[2 * *(unsigned int *)(v92 + 72)];
              if ( (unsigned int *)v26 == v28 )
              {
LABEL_39:
                v82 = 1;
                v32 = i;
              }
              else
              {
                v25 = (__int64 *)((v29 >> 1) & 3);
                while ( 1 )
                {
                  v29 = *(_QWORD *)(*(_QWORD *)v28 + 8LL) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v29 )
                  {
                    v34 = *(_DWORD *)(v29 + 24) | (*(__int64 *)(*(_QWORD *)v28 + 8LL) >> 1) & 3;
                    v29 = (unsigned int)v25 | *(_DWORD *)(v30 + 24);
                    if ( (unsigned int)v29 < v34 )
                    {
                      v35 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v22 + 32) + 152LL)
                                      + 16LL * *(unsigned int *)(v32 + 24)
                                      + 8);
                      v29 = *(_DWORD *)((v35 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v35 >> 1) & 3;
                      if ( v34 < (unsigned int)v29 )
                        break;
                    }
                  }
                  v28 += 2;
                  if ( (unsigned int *)v26 == v28 )
                    goto LABEL_39;
                }
              }
            }
          }
        }
      }
    }
    if ( v90 == ++v23 )
      break;
  }
  v6 = v82;
  if ( !v82 )
    return 0;
  if ( v32 )
  {
    if ( *(_DWORD *)(v32 + 120) <= 1u )
    {
      v91 = v32;
      v36 = sub_2E313E0(v32);
      v37 = v32;
      v38 = (__int64 *)v36;
      if ( v36 == v37 + 48
        || (v39 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL),
            v40 = sub_2DF8360(v39, v36, 0),
            v41 = sub_2E0A0C0(
                    v92,
                    v40 & 0xFFFFFFFFFFFFFFF8LL | 2,
                    *(_QWORD *)(*(_QWORD *)(v39 + 152) + 16LL * *(unsigned int *)(v91 + 24) + 8)),
            v37 = v91,
            !v41) )
      {
        v42 = *(unsigned __int8 **)(v9 + 56);
        v43 = *(_DWORD *)(v92 + 112);
        v44 = *(_QWORD *)(a1 + 32);
        v93 = v42;
        v45 = *(_QWORD *)(v44 + 8) - 800LL;
        if ( v42 )
        {
          v83 = v43;
          v88 = v37;
          sub_B96E90((__int64)&v93, (__int64)v42, 1);
          v37 = v88;
          v43 = v83;
          v94 = (__int64 *)v93;
          if ( v93 )
          {
            sub_B976B0((__int64)&v93, v93, (__int64)&v94);
            v43 = v83;
            v93 = 0;
            v37 = v88;
          }
        }
        else
        {
          v94 = 0;
        }
        v95 = 0;
        v96[0] = 0;
        v46 = sub_2F26260(v37, v38, (__int64 *)&v94, v45, v43);
        v89 = v47;
        v48 = *(_DWORD *)(v85 + 112);
        v97.m128i_i64[0] = 0;
        memset(v98, 0, 24);
        v97.m128i_i32[2] = v48;
        sub_2E8EAD0(v47, (__int64)v46, &v97);
        if ( v94 )
          sub_B91220((__int64)&v94, (__int64)v94);
        if ( v93 )
          sub_B91220((__int64)&v93, (__int64)v93);
        v49 = sub_2E192D0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), v89, 0) & 0xFFFFFFFFFFFFFFF8LL | 4;
        sub_2E0E0B0(v92, v49, (__int64 *)(*(_QWORD *)(a1 + 40) + 56LL), v50, v51, v52);
        v53 = *(_QWORD *)(v92 + 104);
        if ( v53 )
        {
          v54 = v49;
          v55 = *(_QWORD *)(v92 + 104);
          do
          {
            sub_2E0E0B0(v55, v54, (__int64 *)(*(_QWORD *)(a1 + 40) + 56LL), v53, (__int64)v28, v29);
            v55 = *(_QWORD *)(v55 + 104);
          }
          while ( v55 );
          v9 = v81;
        }
        if ( *(_BYTE *)(a1 + 692) )
        {
          v56 = *(__int64 **)(a1 + 672);
          v25 = &v56[*(unsigned int *)(a1 + 684)];
          v26 = *(unsigned int *)(a1 + 684);
          v57 = (__int64 **)v56;
          if ( v56 != v25 )
          {
            while ( (__int64 *)v89 != *v57 )
            {
              if ( v25 == (__int64 *)++v57 )
                goto LABEL_64;
            }
            v26 = (unsigned int)(v26 - 1);
            *(_DWORD *)(a1 + 684) = v26;
            v25 = (__int64 *)v56[v26];
            *v57 = v25;
            ++*(_QWORD *)(a1 + 664);
          }
        }
        else
        {
          v79 = sub_C8CA60(a1 + 664, v89);
          if ( v79 )
          {
            *v79 = -2;
            ++*(_DWORD *)(a1 + 688);
            ++*(_QWORD *)(a1 + 664);
          }
        }
        goto LABEL_64;
      }
    }
    return 0;
  }
LABEL_64:
  v58 = *(_BYTE *)(*(_QWORD *)(v9 + 32) + 44LL);
  sub_2F626D0(a1, v9, v25, v26, (__int64)v28, v29);
  v94 = v96;
  v95 = 0x800000000LL;
  sub_2F65960((__int64)&v97, v92, v86);
  v59 = v97.m128i_i64[1];
  sub_2E18870(*(_QWORD *)(a1 + 40), v92, v84 | 4, (__int64)&v94);
  *(_QWORD *)(v59 + 8) = 0;
  if ( (v58 & 1) != 0 )
  {
    v75 = sub_2F657C0(*(_QWORD *)(a1 + 16), *(_DWORD *)(v92 + 112));
    if ( v75 )
    {
LABEL_83:
      v76 = sub_2DF8360(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), *(_QWORD *)(v75 + 16), 0);
      if ( !(unsigned __int8)sub_2F65890(v92, v76) )
        *(_BYTE *)(v75 + 4) |= 1u;
      while ( 1 )
      {
        v75 = *(_QWORD *)(v75 + 32);
        if ( !v75 )
          break;
        if ( (*(_BYTE *)(v75 + 3) & 0x10) == 0 && (*(_BYTE *)(v75 + 4) & 8) == 0 )
          goto LABEL_83;
      }
    }
  }
  sub_2E12C90(*(_QWORD **)(a1 + 40), v92, v94, (unsigned int)v95, 0, 0);
  if ( *(_QWORD *)(v92 + 104) )
  {
    v80 = v6;
    v63 = *(_QWORD **)(v92 + 104);
    do
    {
      LODWORD(v95) = 0;
      sub_2F65960((__int64)&v97, (__int64)v63, v86);
      v64 = v97.m128i_i64[1];
      sub_2E18870(*(_QWORD *)(a1 + 40), (__int64)v63, v84 | 4, (__int64)&v94);
      v65 = 0;
      v66 = 0;
      *(_QWORD *)(v64 + 8) = 0;
      v67 = (unsigned int)v95;
      if ( (_DWORD)v95 )
      {
        do
        {
          v69 = &v94[v66];
          if ( v84 == (*v69 & 0xFFFFFFFFFFFFFFF8LL) )
          {
            *v69 = v94[v67 - 1];
            v68 = v95 - 1;
            LODWORD(v95) = v95 - 1;
          }
          else
          {
            v68 = v95;
            ++v65;
          }
          v66 = v65;
          v67 = v68;
        }
        while ( v68 != v65 );
      }
      v70 = *(_QWORD **)(a1 + 16);
      v97.m128i_i64[1] = 0x800000000LL;
      v71 = *(_QWORD *)(a1 + 40);
      v97.m128i_i64[0] = (__int64)v98;
      sub_2E0B070(v92, (__int64)&v97, v63[14], v63[15], v70, *(_QWORD *)(v71 + 32));
      sub_2E12C90(*(_QWORD **)(a1 + 40), (__int64)v63, v94, (unsigned int)v95, v97.m128i_i64[0], v97.m128i_u32[2]);
      if ( (_QWORD *)v97.m128i_i64[0] != v98 )
        _libc_free(v97.m128i_u64[0]);
      v63 = (_QWORD *)v63[13];
    }
    while ( v63 );
    v6 = v80;
  }
  if ( (unsigned __int8)sub_2E168A0(*(_QWORD **)(a1 + 40), v92, 0, v60, v61, v62) )
  {
    v78 = *(_QWORD *)(a1 + 40);
    v97.m128i_i64[0] = (__int64)v98;
    v97.m128i_i64[1] = 0x800000000LL;
    sub_2E15100(v78, v92, (__int64)&v97);
    if ( (_QWORD *)v97.m128i_i64[0] != v98 )
      _libc_free(v97.m128i_u64[0]);
  }
  if ( (unsigned __int8)sub_2E168A0(*(_QWORD **)(a1 + 40), v85, 0, v72, v73, v74) )
  {
    v77 = *(_QWORD *)(a1 + 40);
    v97.m128i_i64[0] = (__int64)v98;
    v97.m128i_i64[1] = 0x800000000LL;
    sub_2E15100(v77, v85, (__int64)&v97);
    if ( (_QWORD *)v97.m128i_i64[0] != v98 )
      _libc_free(v97.m128i_u64[0]);
  }
  if ( v94 != v96 )
    _libc_free((unsigned __int64)v94);
  return v6;
}
