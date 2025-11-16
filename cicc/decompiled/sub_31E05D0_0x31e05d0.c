// Function: sub_31E05D0
// Address: 0x31e05d0
//
__int64 *__fastcall sub_31E05D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  bool v5; // r12
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r9
  int v9; // eax
  __int64 v10; // r13
  _QWORD *v11; // r15
  int v12; // ecx
  __int64 v13; // rax
  __int64 *v14; // r15
  unsigned int v15; // edi
  __int64 v16; // rcx
  __int64 v17; // rax
  _DWORD *v18; // rcx
  __int64 *result; // rax
  _QWORD *v20; // rcx
  __int64 v21; // r13
  __int64 v22; // rdi
  __int64 v23; // rbx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  unsigned __int64 v27; // r12
  __int64 v28; // rax
  unsigned __int64 v29; // rcx
  unsigned __int64 v30; // rbx
  __int64 v31; // rsi
  unsigned __int64 v32; // rax
  bool v33; // cf
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // r13
  __int64 v36; // rax
  _QWORD *v37; // rax
  _QWORD *i; // r13
  __int64 v39; // rsi
  __int64 v40; // rsi
  _QWORD *v41; // r12
  void *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdx
  unsigned __int64 v45; // rax
  __int64 v46; // rdx
  int v47; // ecx
  __int64 v48; // rdi
  unsigned int v49; // eax
  int v50; // r11d
  int v51; // eax
  int v52; // eax
  int v53; // eax
  __int64 v54; // rdi
  unsigned int v55; // ecx
  unsigned int v56; // r10d
  unsigned int v57; // r10d
  unsigned __int64 v58; // [rsp+8h] [rbp-78h]
  _QWORD *v59; // [rsp+10h] [rbp-70h]
  unsigned __int64 v60; // [rsp+18h] [rbp-68h]
  _QWORD *v61; // [rsp+20h] [rbp-60h]
  _QWORD *v62; // [rsp+20h] [rbp-60h]
  void *v63; // [rsp+20h] [rbp-60h]
  _QWORD *v64; // [rsp+28h] [rbp-58h]
  _QWORD *v65; // [rsp+28h] [rbp-58h]
  _QWORD *v66; // [rsp+28h] [rbp-58h]
  _QWORD *v67; // [rsp+28h] [rbp-58h]
  unsigned __int64 v68; // [rsp+28h] [rbp-58h]
  _QWORD v69[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v70; // [rsp+40h] [rbp-40h]

  v2 = a1 + 8;
  v69[0] = 0;
  v69[1] = 0;
  v70 = a2;
  v5 = a2 != -8192 && a2 != -4096 && a2 != 0;
  if ( v5 )
  {
    sub_BD73F0((__int64)v69);
    v2 = a1 + 8;
  }
  v6 = *(unsigned int *)(a1 + 32);
  if ( !(_DWORD)v6 )
  {
    ++*(_QWORD *)(a1 + 8);
LABEL_5:
    v6 = (unsigned int)(2 * v6);
    sub_31E01F0(v2, v6);
    v9 = *(_DWORD *)(a1 + 32);
    if ( !v9 )
    {
LABEL_6:
      v10 = v70;
      v11 = 0;
LABEL_7:
      v12 = *(_DWORD *)(a1 + 24) + 1;
      goto LABEL_8;
    }
    v10 = v70;
    v47 = v9 - 1;
    v48 = *(_QWORD *)(a1 + 16);
    v8 = 0;
    v49 = (v9 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
    v11 = (_QWORD *)(v48 + 48LL * v49);
    v7 = 1;
    v6 = v11[2];
    if ( v70 == v6 )
      goto LABEL_7;
    while ( v6 != -4096 )
    {
      if ( v6 == -8192 && !v8 )
        v8 = (__int64)v11;
      v57 = v7 + 1;
      v49 = v47 & (v7 + v49);
      v7 = 48LL * v49;
      v11 = (_QWORD *)(v48 + v7);
      v6 = *(_QWORD *)(v48 + v7 + 16);
      if ( v70 == v6 )
        goto LABEL_7;
      v7 = v57;
    }
LABEL_85:
    if ( v8 )
      v11 = (_QWORD *)v8;
    goto LABEL_7;
  }
  v10 = v70;
  v8 = *(_QWORD *)(a1 + 16);
  v15 = (v6 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
  v7 = v8 + 48LL * v15;
  v16 = *(_QWORD *)(v7 + 16);
  if ( v70 == v16 )
  {
LABEL_19:
    v14 = (__int64 *)(v7 + 24);
    goto LABEL_20;
  }
  v50 = 1;
  v11 = 0;
  while ( v16 != -4096 )
  {
    if ( v16 == -8192 && !v11 )
      v11 = (_QWORD *)v7;
    v15 = (v6 - 1) & (v50 + v15);
    v7 = v8 + 48LL * v15;
    v16 = *(_QWORD *)(v7 + 16);
    if ( v70 == v16 )
      goto LABEL_19;
    ++v50;
  }
  v51 = *(_DWORD *)(a1 + 24);
  if ( !v11 )
    v11 = (_QWORD *)v7;
  ++*(_QWORD *)(a1 + 8);
  v12 = v51 + 1;
  if ( 4 * (v51 + 1) >= (unsigned int)(3 * v6) )
    goto LABEL_5;
  if ( (int)v6 - *(_DWORD *)(a1 + 28) - v12 <= (unsigned int)v6 >> 3 )
  {
    sub_31E01F0(v2, v6);
    v52 = *(_DWORD *)(a1 + 32);
    if ( !v52 )
      goto LABEL_6;
    v10 = v70;
    v53 = v52 - 1;
    v54 = *(_QWORD *)(a1 + 16);
    v8 = 0;
    v55 = v53 & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
    v11 = (_QWORD *)(v54 + 48LL * v55);
    v7 = 1;
    v6 = v11[2];
    if ( v6 == v70 )
      goto LABEL_7;
    while ( v6 != -4096 )
    {
      if ( v6 == -8192 && !v8 )
        v8 = (__int64)v11;
      v56 = v7 + 1;
      v55 = v53 & (v7 + v55);
      v7 = 48LL * v55;
      v11 = (_QWORD *)(v54 + v7);
      v6 = *(_QWORD *)(v54 + v7 + 16);
      if ( v70 == v6 )
        goto LABEL_7;
      v7 = v56;
    }
    goto LABEL_85;
  }
LABEL_8:
  *(_DWORD *)(a1 + 24) = v12;
  if ( v11[2] == -4096 )
  {
    if ( v10 != -4096 )
    {
LABEL_13:
      v11[2] = v10;
      if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
        sub_BD73F0((__int64)v11);
      v10 = v70;
    }
  }
  else
  {
    --*(_DWORD *)(a1 + 28);
    v13 = v11[2];
    if ( v13 != v10 )
    {
      LOBYTE(v6) = v13 != 0;
      if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
        sub_BD60C0(v11);
      goto LABEL_13;
    }
  }
  v11[5] = 0;
  v14 = v11 + 3;
  *(_OWORD *)v14 = 0;
LABEL_20:
  if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
    sub_BD60C0(v69);
  v17 = *v14;
  v18 = (_DWORD *)(*v14 & 0xFFFFFFFFFFFFFFF8LL);
  if ( v18 )
  {
    v6 = v17 >> 2;
    if ( (v17 & 4) == 0 || v18[2] )
    {
      if ( (v6 & 1) != 0 )
        return *(__int64 **)v18;
      return v14;
    }
  }
  v20 = *(_QWORD **)(a1 + 48);
  if ( v20 != *(_QWORD **)(a1 + 56) )
  {
    if ( v20 )
    {
      v20[1] = 2;
      v20[2] = 0;
      v20[3] = a2;
      if ( v5 )
      {
        v64 = v20;
        sub_BD73F0((__int64)(v20 + 1));
        v20 = v64;
      }
      v20[4] = 0;
      *v20 = off_4A35038;
      v20 = *(_QWORD **)(a1 + 48);
    }
    v21 = (__int64)(v20 + 5);
    *(_QWORD *)(a1 + 48) = v20 + 5;
    goto LABEL_35;
  }
  v31 = (__int64)v20 - *(_QWORD *)(a1 + 40);
  v60 = *(_QWORD *)(a1 + 40);
  v8 = 0xCCCCCCCCCCCCCCCDLL * (v31 >> 3);
  if ( v8 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v32 = 1;
  if ( v8 )
    v32 = 0xCCCCCCCCCCCCCCCDLL * (v31 >> 3);
  v33 = __CFADD__(v8, v32);
  v34 = v8 + v32;
  if ( v33 )
  {
    v35 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v34 )
    {
      v58 = 0;
      v21 = 40;
      v59 = 0;
      goto LABEL_56;
    }
    if ( v34 > 0x333333333333333LL )
      v34 = 0x333333333333333LL;
    v35 = 40 * v34;
  }
  v65 = *(_QWORD **)(a1 + 48);
  v36 = sub_22077B0(v35);
  v20 = v65;
  v59 = (_QWORD *)v36;
  v58 = v36 + v35;
  v21 = v36 + 40;
LABEL_56:
  v6 = (__int64)v59 + v31;
  if ( v6 )
  {
    *(_QWORD *)(v6 + 8) = 2;
    *(_QWORD *)(v6 + 16) = 0;
    *(_QWORD *)(v6 + 24) = a2;
    if ( v5 )
    {
      v61 = v20;
      sub_BD73F0(v6 + 8);
      v20 = v61;
    }
    *(_QWORD *)(v6 + 32) = 0;
    *(_QWORD *)v6 = off_4A35038;
  }
  v37 = (_QWORD *)v60;
  if ( v20 != (_QWORD *)v60 )
  {
    for ( i = v59; ; i += 5 )
    {
      if ( i )
      {
        v39 = v37[1];
        i[2] = 0;
        i[1] = v39 & 6;
        v40 = v37[3];
        i[3] = v40;
        if ( v40 != 0 && v40 != -4096 && v40 != -8192 )
        {
          v62 = v20;
          v66 = v37;
          sub_BD6050(i + 1, v37[1] & 0xFFFFFFFFFFFFFFF8LL);
          v20 = v62;
          v37 = v66;
        }
        *i = off_4A35038;
        i[4] = v37[4];
      }
      v37 += 5;
      if ( v20 == v37 )
        break;
    }
    v41 = (_QWORD *)v60;
    v21 = (__int64)(i + 10);
    v42 = &unk_49DB368;
    do
    {
      v6 = v41[3];
      *v41 = v42;
      LOBYTE(v8) = v6 != -4096;
      if ( ((v6 != 0) & (unsigned __int8)v8) != 0 && v6 != -8192 )
      {
        v63 = v42;
        v67 = v20;
        sub_BD60C0(v41 + 1);
        v42 = v63;
        v20 = v67;
      }
      v41 += 5;
    }
    while ( v20 != v41 );
  }
  if ( v60 )
  {
    v6 = *(_QWORD *)(a1 + 56) - v60;
    j_j___libc_free_0(v60);
  }
  *(_QWORD *)(a1 + 48) = v21;
  *(_QWORD *)(a1 + 40) = v59;
  *(_QWORD *)(a1 + 56) = v58;
LABEL_35:
  *(_QWORD *)(v21 - 8) = a1;
  *((_DWORD *)v14 + 4) = -858993459 * ((__int64)(*(_QWORD *)(a1 + 48) - *(_QWORD *)(a1 + 40)) >> 3) - 1;
  v14[1] = *(_QWORD *)(a2 + 72);
  v22 = *(_QWORD *)a1;
  if ( (*(_WORD *)(a2 + 2) & 0x7FFF) == 0 )
  {
    v23 = sub_E6C430(v22, v6, v2, (__int64)v20, v7);
    v26 = *v14;
    v27 = *v14 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v27 )
      goto LABEL_37;
LABEL_45:
    v30 = v23 & 0xFFFFFFFFFFFFFFFBLL;
    *v14 = v30;
    v29 = v30;
    goto LABEL_41;
  }
  v23 = sub_E6C270(v22, v6, v2, (__int64)v20, v7, v8);
  v26 = *v14;
  v27 = *v14 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v27 )
    goto LABEL_45;
LABEL_37:
  if ( (v26 & 4) == 0 )
  {
    v43 = sub_22077B0(0x30u);
    if ( v43 )
    {
      *(_QWORD *)v43 = v43 + 16;
      *(_QWORD *)(v43 + 8) = 0x400000000LL;
    }
    v44 = v43;
    v45 = v43 & 0xFFFFFFFFFFFFFFF8LL;
    *v14 = v44 | 4;
    v46 = *(unsigned int *)(v45 + 8);
    v25 = v46 + 1;
    if ( v46 + 1 > (unsigned __int64)*(unsigned int *)(v45 + 12) )
    {
      v68 = v45;
      sub_C8D5F0(v45, (const void *)(v45 + 16), v46 + 1, 8u, v24, v25);
      v45 = v68;
      v46 = *(unsigned int *)(v68 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v45 + 8 * v46) = v27;
    ++*(_DWORD *)(v45 + 8);
    v27 = *v14 & 0xFFFFFFFFFFFFFFF8LL;
  }
  v28 = *(unsigned int *)(v27 + 8);
  if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(v27 + 12) )
  {
    sub_C8D5F0(v27, (const void *)(v27 + 16), v28 + 1, 8u, v24, v25);
    v28 = *(unsigned int *)(v27 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v27 + 8 * v28) = v23;
  ++*(_DWORD *)(v27 + 8);
  v29 = *v14;
LABEL_41:
  result = 0;
  if ( (v29 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    if ( (v29 & 4) != 0 )
      return *(__int64 **)(v29 & 0xFFFFFFFFFFFFFFF8LL);
    return v14;
  }
  return result;
}
