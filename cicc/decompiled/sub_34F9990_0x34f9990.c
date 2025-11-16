// Function: sub_34F9990
// Address: 0x34f9990
//
__int64 *__fastcall sub_34F9990(__int64 a1, __int64 a2, __int32 a3, unsigned int a4)
{
  int v4; // r9d
  unsigned int v5; // eax
  __int64 v6; // r15
  __int64 v10; // r14
  unsigned __int64 v11; // rdx
  __int64 *v12; // r8
  unsigned int v13; // esi
  __int64 v14; // r9
  int v15; // r15d
  unsigned int v16; // edi
  __int32 *v17; // rax
  int v18; // ecx
  __int64 *v19; // r15
  __int64 v20; // rsi
  __int64 v21; // r9
  unsigned __int64 v22; // rax
  __int64 i; // rdi
  __int16 v24; // dx
  __int64 v25; // rdi
  __int64 v26; // r8
  unsigned int v27; // esi
  __int64 *v28; // rdx
  __int64 v29; // r10
  unsigned __int64 v30; // r14
  __int64 *v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 *v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rdi
  __int64 *result; // rax
  unsigned int v40; // eax
  __int64 v41; // rcx
  __int64 *v42; // r15
  __int64 v43; // rax
  int v44; // edx
  __int64 v45; // r8
  unsigned __int64 v46; // r10
  __int64 *v47; // rax
  __int64 *v48; // rsi
  int v49; // edx
  int v50; // ecx
  __int64 *v51; // rax
  __int64 v52; // rcx
  unsigned __int64 *v53; // r14
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // r15
  unsigned __int64 v56; // rdi
  unsigned __int64 v57; // rdi
  int v58; // ecx
  int v59; // edi
  int v60; // edi
  __int64 v61; // r9
  unsigned int v62; // eax
  __int64 v63; // r15
  int v64; // r11d
  __int32 *v65; // r10
  int v66; // esi
  __int32 *v67; // rdx
  int v68; // edi
  int v69; // edi
  __int64 v70; // r10
  unsigned int v71; // edx
  __int64 v72; // r15
  int v73; // r9d
  __int32 *v74; // r11
  int v75; // esi
  __int32 *v76; // rax
  __int64 *v77; // [rsp+8h] [rbp-58h]
  __int64 v79; // [rsp+10h] [rbp-50h]
  unsigned __int64 v80; // [rsp+10h] [rbp-50h]
  __int64 *v81; // [rsp+10h] [rbp-50h]
  __int64 *v82; // [rsp+10h] [rbp-50h]
  unsigned __int64 v83; // [rsp+10h] [rbp-50h]
  __int64 v84; // [rsp+18h] [rbp-48h]
  int *v85; // [rsp+18h] [rbp-48h]
  __int32 *v86; // [rsp+18h] [rbp-48h]
  __int64 v87; // [rsp+18h] [rbp-48h]
  __m128i v88; // [rsp+20h] [rbp-40h] BYREF

  v4 = a4;
  v5 = a4 & 0x7FFFFFFF;
  v6 = 8LL * (a4 & 0x7FFFFFFF);
  v10 = *(_QWORD *)(a1 + 16);
  v11 = *(unsigned int *)(v10 + 160);
  if ( (a4 & 0x7FFFFFFF) >= (unsigned int)v11 || (v12 = *(__int64 **)(*(_QWORD *)(v10 + 152) + 8LL * v5)) == 0 )
  {
    v40 = v5 + 1;
    if ( (unsigned int)v11 < v40 && v40 != v11 )
    {
      if ( v40 >= v11 )
      {
        v45 = *(_QWORD *)(v10 + 168);
        v46 = v40 - v11;
        if ( v40 > (unsigned __int64)*(unsigned int *)(v10 + 164) )
        {
          v83 = v40 - v11;
          v87 = *(_QWORD *)(v10 + 168);
          sub_C8D5F0(v10 + 152, (const void *)(v10 + 168), v40, 8u, v45, a4);
          v11 = *(unsigned int *)(v10 + 160);
          v4 = a4;
          v46 = v83;
          v45 = v87;
        }
        v41 = *(_QWORD *)(v10 + 152);
        v47 = (__int64 *)(v41 + 8 * v11);
        v48 = &v47[v46];
        if ( v47 != v48 )
        {
          do
            *v47++ = v45;
          while ( v48 != v47 );
          LODWORD(v11) = *(_DWORD *)(v10 + 160);
          v41 = *(_QWORD *)(v10 + 152);
        }
        *(_DWORD *)(v10 + 160) = v46 + v11;
        goto LABEL_28;
      }
      *(_DWORD *)(v10 + 160) = v40;
    }
    v41 = *(_QWORD *)(v10 + 152);
LABEL_28:
    v42 = (__int64 *)(v41 + v6);
    v43 = sub_2E10F30(v4);
    *v42 = v43;
    v84 = v43;
    sub_2E11E80((_QWORD *)v10, v43);
    v12 = (__int64 *)v84;
  }
  v13 = *(_DWORD *)(a1 + 256);
  if ( v13 )
  {
    v14 = *(_QWORD *)(a1 + 240);
    v15 = 37 * a3;
    v16 = (v13 - 1) & (37 * a3);
    v17 = (__int32 *)(v14 + 16LL * v16);
    v18 = *v17;
    if ( *v17 == a3 )
    {
LABEL_5:
      v19 = (__int64 *)*((_QWORD *)v17 + 1);
      goto LABEL_6;
    }
    v85 = 0;
    v49 = 1;
    while ( v18 != 0x7FFFFFFF )
    {
      if ( v18 == 0x80000000 )
      {
        if ( v85 )
          v17 = v85;
        v85 = v17;
      }
      v16 = (v13 - 1) & (v49 + v16);
      v17 = (__int32 *)(v14 + 16LL * v16);
      v18 = *v17;
      if ( *v17 == a3 )
        goto LABEL_5;
      ++v49;
    }
    if ( v85 )
      v17 = v85;
    ++*(_QWORD *)(a1 + 232);
    v86 = v17;
    v50 = *(_DWORD *)(a1 + 248) + 1;
    if ( 4 * v50 < 3 * v13 )
    {
      if ( v13 - *(_DWORD *)(a1 + 252) - v50 > v13 >> 3 )
        goto LABEL_50;
      v82 = v12;
      sub_34F6A50(a1 + 232, v13);
      v68 = *(_DWORD *)(a1 + 256);
      if ( v68 )
      {
        v69 = v68 - 1;
        v70 = *(_QWORD *)(a1 + 240);
        v12 = v82;
        v71 = v69 & v15;
        v72 = 16LL * (v69 & (unsigned int)v15);
        v73 = *(_DWORD *)(v70 + v72);
        v86 = (__int32 *)(v70 + v72);
        v50 = *(_DWORD *)(a1 + 248) + 1;
        if ( v73 != a3 )
        {
          v74 = (__int32 *)(v70 + v72);
          v75 = 1;
          v76 = 0;
          while ( v73 != 0x7FFFFFFF )
          {
            if ( !v76 && v73 == 0x80000000 )
              v76 = v74;
            v71 = v69 & (v75 + v71);
            v74 = (__int32 *)(v70 + 16LL * v71);
            v73 = *v74;
            if ( *v74 == a3 )
            {
              v86 = (__int32 *)(v70 + 16LL * v71);
              goto LABEL_50;
            }
            ++v75;
          }
          if ( !v76 )
            v76 = v74;
          v86 = v76;
        }
        goto LABEL_50;
      }
LABEL_99:
      ++*(_DWORD *)(a1 + 248);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 232);
  }
  v81 = v12;
  sub_34F6A50(a1 + 232, 2 * v13);
  v59 = *(_DWORD *)(a1 + 256);
  if ( !v59 )
    goto LABEL_99;
  v60 = v59 - 1;
  v61 = *(_QWORD *)(a1 + 240);
  v12 = v81;
  v62 = v60 & (37 * a3);
  v63 = 16LL * v62;
  v64 = *(_DWORD *)(v61 + v63);
  v86 = (__int32 *)(v61 + v63);
  v50 = *(_DWORD *)(a1 + 248) + 1;
  if ( v64 != a3 )
  {
    v65 = (__int32 *)(v61 + v63);
    v66 = 1;
    v67 = 0;
    while ( v64 != 0x7FFFFFFF )
    {
      if ( !v67 && v64 == 0x80000000 )
        v67 = v65;
      v62 = v60 & (v66 + v62);
      v65 = (__int32 *)(v61 + 16LL * v62);
      v64 = *v65;
      if ( *v65 == a3 )
      {
        v86 = (__int32 *)(v61 + 16LL * v62);
        goto LABEL_50;
      }
      ++v66;
    }
    if ( !v67 )
      v67 = v65;
    v86 = v67;
  }
LABEL_50:
  *(_DWORD *)(a1 + 248) = v50;
  if ( *v86 != 0x7FFFFFFF )
    --*(_DWORD *)(a1 + 252);
  v77 = v12;
  *v86 = a3;
  *((_QWORD *)v86 + 1) = 0;
  v79 = v12[14];
  v51 = (__int64 *)sub_22077B0(0x78u);
  v19 = v51;
  if ( v51 )
  {
    v51[12] = 0;
    *v51 = (__int64)(v51 + 2);
    v51[1] = 0x200000000LL;
    v51[8] = (__int64)(v51 + 10);
    v51[9] = 0x200000000LL;
    v51[13] = 0;
    v51[14] = v79;
  }
  sub_2F68500((__int64)v51, v77, (__int64 *)(v10 + 56), v52, (__int64)v77);
  v53 = (unsigned __int64 *)*((_QWORD *)v86 + 1);
  *((_QWORD *)v86 + 1) = v19;
  if ( v53 )
  {
    sub_2E0AFD0((__int64)v53);
    v54 = v53[12];
    v80 = v54;
    if ( v54 )
    {
      v55 = *(_QWORD *)(v54 + 16);
      while ( v55 )
      {
        sub_34F51B0(*(_QWORD *)(v55 + 24));
        v56 = v55;
        v55 = *(_QWORD *)(v55 + 16);
        j_j___libc_free_0(v56);
      }
      j_j___libc_free_0(v80);
    }
    v57 = v53[8];
    if ( (unsigned __int64 *)v57 != v53 + 10 )
      _libc_free(v57);
    if ( (unsigned __int64 *)*v53 != v53 + 2 )
      _libc_free(*v53);
    j_j___libc_free_0((unsigned __int64)v53);
    v17 = v86;
    goto LABEL_5;
  }
LABEL_6:
  v20 = a2;
  v21 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL);
  v22 = a2;
  if ( (*(_DWORD *)(a2 + 44) & 4) != 0 )
  {
    do
      v22 = *(_QWORD *)v22 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v22 + 44) & 4) != 0 );
  }
  if ( (*(_DWORD *)(a2 + 44) & 8) != 0 )
  {
    do
      v20 = *(_QWORD *)(v20 + 8);
    while ( (*(_BYTE *)(v20 + 44) & 8) != 0 );
  }
  for ( i = *(_QWORD *)(v20 + 8); i != v22; v22 = *(_QWORD *)(v22 + 8) )
  {
    v24 = *(_WORD *)(v22 + 68);
    if ( (unsigned __int16)(v24 - 14) > 4u && v24 != 24 )
      break;
  }
  v25 = *(unsigned int *)(v21 + 144);
  v26 = *(_QWORD *)(v21 + 128);
  if ( !(_DWORD)v25 )
  {
LABEL_36:
    v28 = (__int64 *)(v26 + 16 * v25);
    goto LABEL_16;
  }
  v27 = (v25 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
  v28 = (__int64 *)(v26 + 16LL * v27);
  v29 = *v28;
  if ( *v28 != v22 )
  {
    v44 = 1;
    while ( v29 != -4096 )
    {
      v58 = v44 + 1;
      v27 = (v25 - 1) & (v44 + v27);
      v28 = (__int64 *)(v26 + 16LL * v27);
      v29 = *v28;
      if ( *v28 == v22 )
        goto LABEL_16;
      v44 = v58;
    }
    goto LABEL_36;
  }
LABEL_16:
  v30 = v28[1] & 0xFFFFFFFFFFFFFFF8LL;
  v31 = (__int64 *)sub_2E09D00(v19, v30 | 4);
  if ( v31 == (__int64 *)(*v19 + 24LL * *((unsigned int *)v19 + 2))
    || (*(_DWORD *)((*v31 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v31 >> 1) & 3) > (*(_DWORD *)(v30 + 24) | 2u) )
  {
    v32 = 0;
  }
  else
  {
    v32 = v31[2];
  }
  v88.m128i_i32[0] = a3;
  v88.m128i_i64[1] = v32;
  v33 = sub_34F9630(a1 + 264, &v88);
  v38 = v33;
  if ( !*(_BYTE *)(v33 + 28) )
    return sub_C8CC70(v38, a2, (__int64)v34, v35, v36, v37);
  result = *(__int64 **)(v33 + 8);
  v35 = *(unsigned int *)(v38 + 20);
  v34 = &result[v35];
  if ( result == v34 )
  {
LABEL_29:
    if ( (unsigned int)v35 >= *(_DWORD *)(v38 + 16) )
      return sub_C8CC70(v38, a2, (__int64)v34, v35, v36, v37);
    *(_DWORD *)(v38 + 20) = v35 + 1;
    *v34 = a2;
    ++*(_QWORD *)v38;
  }
  else
  {
    while ( a2 != *result )
    {
      if ( v34 == ++result )
        goto LABEL_29;
    }
  }
  return result;
}
