// Function: sub_1FE9790
// Address: 0x1fe9790
//
__int64 __fastcall sub_1FE9790(
        __int64 *a1,
        unsigned __int64 a2,
        unsigned int a3,
        unsigned __int8 a4,
        unsigned __int8 a5,
        signed int a6,
        __int64 a7)
{
  unsigned int v7; // r15d
  __int64 v9; // rax
  __int64 v10; // r10
  unsigned int v11; // r9d
  __int64 v12; // rax
  int v13; // ebx
  char v14; // al
  __int64 v15; // r12
  __int64 v16; // r14
  __int64 v17; // rbx
  char v18; // r11
  unsigned int v19; // r15d
  __int64 v20; // rcx
  __int64 v21; // rax
  char v22; // r9
  __int16 v23; // ax
  __int64 v24; // rdi
  __int64 v25; // rsi
  unsigned int v26; // edx
  _QWORD *v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // rax
  __int64 v30; // r8
  int v31; // r9d
  __int64 *v32; // r12
  __int64 v33; // rdi
  __int64 (__fastcall *v34)(__int64, unsigned __int8); // rax
  __int32 v35; // ebx
  __int32 v36; // eax
  __int64 *v37; // r12
  __int64 v38; // rsi
  __int64 v39; // r14
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rdx
  unsigned int v43; // esi
  unsigned int v44; // ecx
  int v45; // r9d
  __int64 v46; // r8
  unsigned int v47; // edi
  __int64 result; // rax
  __int64 v49; // r14
  unsigned int v50; // edi
  int v51; // r9d
  unsigned int j; // eax
  __int64 v53; // rdi
  unsigned int v54; // eax
  int v55; // eax
  int v56; // ecx
  int v57; // r8d
  unsigned int i; // eax
  __int64 v59; // rdx
  unsigned int v60; // eax
  __int64 v61; // r8
  int v62; // r9d
  __int64 v63; // rdi
  __int64 (__fastcall *v64)(__int64, unsigned __int8); // rax
  int v65; // esi
  __int64 v66; // r8
  int v67; // esi
  int v68; // edi
  __int64 v69; // rcx
  unsigned int v70; // edx
  unsigned int v71; // edx
  int v72; // r10d
  int v73; // edx
  int v74; // edx
  int v75; // edx
  __int64 v76; // rdi
  int v77; // ecx
  unsigned int k; // r12d
  unsigned int v79; // r12d
  int v80; // esi
  int v81; // r9d
  __int64 v82; // [rsp+8h] [rbp-B8h]
  char v83; // [rsp+16h] [rbp-AAh]
  unsigned __int8 v84; // [rsp+17h] [rbp-A9h]
  _QWORD *v85; // [rsp+18h] [rbp-A8h]
  __int64 v86; // [rsp+20h] [rbp-A0h]
  char v89; // [rsp+30h] [rbp-90h]
  __int64 v90; // [rsp+30h] [rbp-90h]
  __int64 v93; // [rsp+38h] [rbp-88h]
  unsigned __int64 v94; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v95; // [rsp+48h] [rbp-78h]
  signed int v96[4]; // [rsp+50h] [rbp-70h] BYREF
  __m128i v97; // [rsp+60h] [rbp-60h] BYREF
  __int64 v98; // [rsp+70h] [rbp-50h]
  __int64 v99; // [rsp+78h] [rbp-48h]
  __int64 v100; // [rsp+80h] [rbp-40h]

  v7 = a3;
  if ( a6 < 0 )
  {
    if ( a4 )
    {
      v55 = *(_DWORD *)(a7 + 24);
      if ( v55 )
      {
        v56 = v55 - 1;
        v57 = 1;
        for ( i = (v55 - 1) & (a3 + ((a2 >> 9) ^ (a2 >> 4))); ; i = v56 & v60 )
        {
          v59 = *(_QWORD *)(a7 + 8) + 24LL * i;
          if ( a2 == *(_QWORD *)v59 && v7 == *(_DWORD *)(v59 + 8) )
            break;
          if ( !*(_QWORD *)v59 && *(_DWORD *)(v59 + 8) == -1 )
            goto LABEL_46;
          v60 = v57 + i;
          ++v57;
        }
        *(_QWORD *)v59 = 0;
        *(_DWORD *)(v59 + 8) = -2;
        --*(_DWORD *)(a7 + 16);
        ++*(_DWORD *)(a7 + 20);
      }
    }
LABEL_46:
    v94 = a2;
    v95 = v7;
    v96[0] = a6;
    return sub_1FE7CB0((__int64)&v97, a7, &v94, v96);
  }
  v9 = *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * a3);
  v84 = v9;
  if ( (_BYTE)v9 && (v33 = a1[4], (v86 = *(_QWORD *)(v33 + 8 * v9 + 120)) != 0) )
  {
    v34 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v33 + 288LL);
    if ( v34 == sub_1D45FB0 )
    {
      if ( a4 || a5 || (v10 = *(_QWORD *)(a2 + 48)) == 0 )
      {
        v32 = sub_1F4ABE0(a1[3], a6, v84);
        goto LABEL_35;
      }
      goto LABEL_6;
    }
    v86 = v34(v33, v84);
    if ( a4 || a5 || (v10 = *(_QWORD *)(a2 + 48)) == 0 )
    {
      v32 = sub_1F4ABE0(a1[3], a6, v84);
      if ( v86 )
      {
LABEL_35:
        v35 = a6;
        if ( *(char *)(*v32 + 28) < 0 )
          goto LABEL_38;
LABEL_36:
        v36 = sub_1E6B9A0(a1[1], v86, (unsigned __int8 *)byte_3F871B3, 0, v30, v31);
        goto LABEL_37;
      }
      goto LABEL_72;
    }
  }
  else
  {
    v89 = a4 | a5;
    if ( a4 | a5 )
    {
      v32 = sub_1F4ABE0(a1[3], a6, v9);
      goto LABEL_66;
    }
    v10 = *(_QWORD *)(a2 + 48);
    if ( !v10 )
    {
      v32 = sub_1F4ABE0(a1[3], a6, v9);
LABEL_72:
      v89 = 1;
LABEL_66:
      v63 = a1[4];
      v64 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v63 + 288LL);
      if ( v64 == sub_1D45FB0 )
        v86 = *(_QWORD *)(v63 + 8LL * v84 + 120);
      else
        v86 = v64(v63, v84);
LABEL_29:
      if ( !v89 )
        goto LABEL_36;
      goto LABEL_35;
    }
    v86 = 0;
  }
LABEL_6:
  v89 = 1;
  v11 = v7;
  v85 = (_QWORD *)a1[3];
  while ( 1 )
  {
    v15 = *(_QWORD *)(v10 + 16);
    if ( *(_WORD *)(v15 + 24) == 46 )
    {
      v12 = *(_QWORD *)(v15 + 32);
      if ( a2 == *(_QWORD *)(v12 + 80) && v11 == *(_DWORD *)(v12 + 88) )
        break;
    }
    v16 = *(unsigned int *)(v15 + 56);
    v17 = 0;
    v18 = 1;
    v19 = v11;
    if ( (_DWORD)v16 )
    {
      do
      {
        while ( 1 )
        {
          v20 = *(_QWORD *)(v15 + 32) + 40 * v17;
          v21 = *(unsigned int *)(v20 + 8);
          if ( *(_QWORD *)v20 != a2 )
            goto LABEL_17;
          if ( (_DWORD)v21 != v19 )
            goto LABEL_17;
          v22 = *(_BYTE *)(*(_QWORD *)(a2 + 40) + 16 * v21);
          if ( v22 == 111 || v22 == 1 )
            goto LABEL_17;
          v23 = *(_WORD *)(v15 + 24);
          v18 = 0;
          if ( v23 >= 0 )
            goto LABEL_17;
          v24 = a1[2];
          v25 = *(_QWORD *)(v24 + 8) + ((__int64)~v23 << 6);
          v26 = *(unsigned __int8 *)(v25 + 4) + (_DWORD)v17;
          if ( v26 >= *(unsigned __int16 *)(v25 + 2) )
            goto LABEL_17;
          v82 = v10;
          v83 = v22;
          v27 = (_QWORD *)sub_1F3AD60(v24, v25, v26, v85, *a1);
          v28 = sub_1F4AAF0((__int64)v85, v27);
          v10 = v82;
          v18 = 0;
          if ( !v86 )
          {
            v86 = (__int64)v28;
            v85 = (_QWORD *)a1[3];
            goto LABEL_17;
          }
          v85 = (_QWORD *)a1[3];
          if ( v28 )
            break;
LABEL_17:
          if ( ++v17 == v16 )
            goto LABEL_27;
        }
        v29 = sub_1F4AF90(a1[3], v86, (__int64)v28, v83);
        v10 = v82;
        v18 = 0;
        if ( !v29 )
        {
          v85 = (_QWORD *)a1[3];
          goto LABEL_17;
        }
        v86 = v29;
        ++v17;
        v85 = (_QWORD *)a1[3];
      }
      while ( v17 != v16 );
LABEL_27:
      v10 = *(_QWORD *)(v10 + 32);
      v89 &= v18;
      v11 = v19;
      if ( !v10 )
      {
LABEL_28:
        v7 = v11;
        v32 = sub_1F4ABE0((__int64)v85, a6, v84);
        if ( v86 )
          goto LABEL_29;
        goto LABEL_66;
      }
    }
    else
    {
LABEL_13:
      v10 = *(_QWORD *)(v10 + 32);
      if ( !v10 )
        goto LABEL_28;
    }
  }
  v13 = *(_DWORD *)(*(_QWORD *)(v12 + 40) + 84LL);
  if ( v13 >= 0 )
  {
    v14 = v89;
    if ( v13 != a6 )
      v14 = 0;
    v89 = v14;
    goto LABEL_13;
  }
  v7 = v11;
  sub_1F4ABE0((__int64)v85, a6, v84);
  v36 = sub_1E6B9A0(
          a1[1],
          *(_QWORD *)(*(_QWORD *)(a1[1] + 24) + 16LL * (v13 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
          (unsigned __int8 *)byte_3F871B3,
          0,
          v61,
          v62);
LABEL_37:
  v35 = v36;
  v37 = (__int64 *)a1[6];
  v90 = a1[5];
  v38 = *(_QWORD *)(a1[2] + 8);
  v93 = *(_QWORD *)(v90 + 56);
  v39 = (__int64)sub_1E0B640(v93, v38 + 960, (__int64 *)(a2 + 72), 0);
  sub_1DD5BA0((__int64 *)(v90 + 16), v39);
  v40 = *v37;
  v41 = *(_QWORD *)v39;
  *(_QWORD *)(v39 + 8) = v37;
  v40 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v39 = v40 | v41 & 7;
  *(_QWORD *)(v40 + 8) = v39;
  *v37 = v39 | *v37 & 7;
  v97.m128i_i64[0] = 0x10000000;
  v98 = 0;
  v97.m128i_i32[2] = v35;
  v99 = 0;
  v100 = 0;
  sub_1E1A9C0(v39, v93, &v97);
  v97.m128i_i64[0] = 0;
  v98 = 0;
  v97.m128i_i32[2] = a6;
  v99 = 0;
  v100 = 0;
  sub_1E1A9C0(v39, v93, &v97);
LABEL_38:
  v42 = *(_QWORD *)(a7 + 8);
  v43 = *(_DWORD *)(a7 + 24);
  if ( !a4 )
    goto LABEL_39;
  if ( !v43 )
    goto LABEL_73;
  v44 = v43 - 1;
  v51 = 1;
  for ( j = (v43 - 1) & (v7 + ((a2 >> 9) ^ (a2 >> 4))); ; j = v44 & v54 )
  {
    v53 = v42 + 24LL * j;
    if ( a2 == *(_QWORD *)v53 )
      break;
    if ( !*(_QWORD *)v53 && *(_DWORD *)(v53 + 8) == -1 )
      goto LABEL_41;
LABEL_51:
    v54 = v51 + j;
    ++v51;
  }
  if ( v7 != *(_DWORD *)(v53 + 8) )
    goto LABEL_51;
  *(_QWORD *)v53 = 0;
  *(_DWORD *)(v53 + 8) = -2;
  --*(_DWORD *)(a7 + 16);
  v42 = *(_QWORD *)(a7 + 8);
  ++*(_DWORD *)(a7 + 20);
  v43 = *(_DWORD *)(a7 + 24);
LABEL_39:
  if ( !v43 )
  {
LABEL_73:
    v43 = 0;
    ++*(_QWORD *)a7;
    goto LABEL_74;
  }
  v44 = v43 - 1;
LABEL_41:
  v45 = 1;
  v46 = 0;
  v47 = v44 & (v7 + ((a2 >> 9) ^ (a2 >> 4)));
  while ( 2 )
  {
    result = v42 + 24LL * v47;
    v49 = *(_QWORD *)result;
    if ( a2 == *(_QWORD *)result )
    {
      if ( v7 == *(_DWORD *)(result + 8) )
        return result;
      goto LABEL_44;
    }
    if ( v49 )
    {
LABEL_44:
      v50 = v45 + v47;
      ++v45;
      v47 = v44 & v50;
      continue;
    }
    break;
  }
  v72 = *(_DWORD *)(result + 8);
  if ( v72 != -1 )
  {
    if ( !v46 && v72 == -2 )
      v46 = v42 + 24LL * v47;
    goto LABEL_44;
  }
  if ( v46 )
    result = v46;
  ++*(_QWORD *)a7;
  v73 = *(_DWORD *)(a7 + 16) + 1;
  if ( 4 * v73 < 3 * v43 )
  {
    if ( v43 - (v73 + *(_DWORD *)(a7 + 20)) > v43 >> 3 )
      goto LABEL_97;
    sub_1FE7AA0(a7, v43);
    v74 = *(_DWORD *)(a7 + 24);
    if ( v74 )
    {
      v75 = v74 - 1;
      v76 = *(_QWORD *)(a7 + 8);
      v77 = 1;
      for ( k = v75 & (v7 + ((a2 >> 9) ^ (a2 >> 4))); ; k = v75 & v79 )
      {
        result = v76 + 24LL * k;
        if ( a2 == *(_QWORD *)result )
        {
          if ( v7 == *(_DWORD *)(result + 8) )
            goto LABEL_108;
        }
        else if ( !*(_QWORD *)result )
        {
          v80 = *(_DWORD *)(result + 8);
          if ( v80 == -1 )
          {
            if ( v49 )
              result = v49;
            v73 = *(_DWORD *)(a7 + 16) + 1;
            goto LABEL_97;
          }
          if ( !v49 && v80 == -2 )
            v49 = v76 + 24LL * k;
        }
        v79 = v77 + k;
        ++v77;
      }
    }
LABEL_128:
    ++*(_DWORD *)(a7 + 16);
    BUG();
  }
LABEL_74:
  sub_1FE7AA0(a7, 2 * v43);
  v65 = *(_DWORD *)(a7 + 24);
  if ( !v65 )
    goto LABEL_128;
  v66 = *(_QWORD *)(a7 + 8);
  v67 = v65 - 1;
  v68 = 1;
  v69 = 0;
  v70 = v67 & (v7 + ((a2 >> 9) ^ (a2 >> 4)));
  while ( 2 )
  {
    result = v66 + 24LL * v70;
    if ( a2 == *(_QWORD *)result )
    {
      if ( v7 == *(_DWORD *)(result + 8) )
      {
LABEL_108:
        v73 = *(_DWORD *)(a7 + 16) + 1;
        goto LABEL_97;
      }
      goto LABEL_78;
    }
    if ( *(_QWORD *)result )
    {
LABEL_78:
      v71 = v68 + v70;
      ++v68;
      v70 = v67 & v71;
      continue;
    }
    break;
  }
  v81 = *(_DWORD *)(result + 8);
  if ( v81 != -1 )
  {
    if ( v81 == -2 && !v69 )
      v69 = v66 + 24LL * v70;
    goto LABEL_78;
  }
  if ( v69 )
    result = v69;
  v73 = *(_DWORD *)(a7 + 16) + 1;
LABEL_97:
  *(_DWORD *)(a7 + 16) = v73;
  if ( *(_QWORD *)result || *(_DWORD *)(result + 8) != -1 )
    --*(_DWORD *)(a7 + 20);
  *(_QWORD *)result = a2;
  *(_DWORD *)(result + 8) = v7;
  *(_DWORD *)(result + 16) = v35;
  return result;
}
