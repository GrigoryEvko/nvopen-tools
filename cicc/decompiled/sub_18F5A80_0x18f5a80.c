// Function: sub_18F5A80
// Address: 0x18f5a80
//
__int64 __fastcall sub_18F5A80(
        __int64 *a1,
        __int64 *a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 *v10; // rbx
  __int64 *v11; // r15
  unsigned __int16 *v13; // r14
  __int64 v14; // rax
  unsigned __int16 *v15; // r13
  unsigned __int16 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r13
  unsigned __int64 v20; // rsi
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // rcx
  unsigned int v23; // esi
  __int64 v24; // rdx
  __int64 v25; // r9
  unsigned int v26; // r10d
  __int64 *v27; // r12
  __int64 v28; // rcx
  _QWORD *v29; // rdx
  __int64 v30; // r14
  _QWORD *v31; // r9
  _QWORD *v32; // rcx
  __int64 v33; // rcx
  __int64 result; // rax
  unsigned __int64 v35; // rsi
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // r8
  __int64 v42; // r13
  __int64 *v43; // rbx
  __int64 i; // r12
  __int64 v45; // r15
  __int64 v46; // rax
  __int64 v47; // r8
  _QWORD *v48; // rax
  __int64 v49; // rdx
  _BOOL8 v50; // rdi
  __int64 v51; // rax
  int v52; // r14d
  __int64 *v53; // r11
  __int64 v54; // rax
  int v55; // eax
  __int64 v56; // rcx
  __int64 *v57; // [rsp+0h] [rbp-C0h]
  _QWORD *v58; // [rsp+8h] [rbp-B8h]
  __int64 v59; // [rsp+8h] [rbp-B8h]
  __int64 v61; // [rsp+10h] [rbp-B0h]
  __int64 v62; // [rsp+10h] [rbp-B0h]
  _QWORD *v63; // [rsp+10h] [rbp-B0h]
  _QWORD *v64; // [rsp+18h] [rbp-A8h]
  __int64 v65; // [rsp+18h] [rbp-A8h]
  _QWORD *v66; // [rsp+18h] [rbp-A8h]
  _QWORD v69[6]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v70[12]; // [rsp+60h] [rbp-60h] BYREF

  if ( a1[1] == -1 )
    return 4;
  v10 = a2;
  if ( a2[1] == -1 )
    return 4;
  v11 = a1;
  v13 = (unsigned __int16 *)sub_1649C60(*a2);
  v14 = sub_1649C60(*a1);
  v15 = (unsigned __int16 *)v14;
  if ( v13 == (unsigned __int16 *)v14
    || (v70[0] = v14,
        v70[1] = 1,
        memset(&v70[2], 0, 24),
        v69[0] = v13,
        v69[1] = 1,
        memset(&v69[2], 0, 24),
        (unsigned __int8)sub_134CB50(a9, (__int64)v69, (__int64)v70) == 3) )
  {
    if ( a1[1] >= (unsigned __int64)a2[1] )
      return 1;
  }
  v64 = (_QWORD *)sub_14AD280((__int64)v13, a3, 6u);
  if ( v64 != (_QWORD *)sub_14AD280((__int64)v15, a3, 6u) )
    return 4;
  LOWORD(v69[0]) = 0;
  BYTE2(v69[0]) = sub_15E4690(a10, 0);
  if ( (unsigned __int8)sub_140E950(v64, v70, a3, a4, v69[0])
    && v70[0] != -1
    && a1[1] == v70[0]
    && (unsigned __int64)a2[1] <= v70[0] )
  {
    return 1;
  }
  *a5 = 0;
  *a6 = 0;
  v16 = sub_14AC610(v13, a5, a3);
  if ( v16 != sub_14AC610(v15, a6, a3) )
    return 4;
  v17 = *a5;
  v18 = *a6;
  v19 = *a6;
  if ( *a5 >= *a6 )
  {
    v20 = a1[1];
    v21 = v10[1];
    if ( v20 >= v21 && v20 >= v17 - v18 + v21 )
      return 1;
  }
  if ( !byte_4FAE4E0 )
  {
    if ( !byte_4FAE400 || v17 > v18 )
      goto LABEL_22;
    goto LABEL_41;
  }
  v22 = v10[1];
  if ( v18 >= (__int64)(v17 + v22) || (v65 = v18 + a1[1], v17 > v65) )
  {
    if ( !byte_4FAE400 || v17 > v18 )
      return 4;
    goto LABEL_36;
  }
  v23 = *(_DWORD *)(a8 + 24);
  if ( !v23 )
  {
    ++*(_QWORD *)a8;
    goto LABEL_91;
  }
  v24 = a7;
  v25 = *(_QWORD *)(a8 + 8);
  v26 = (v23 - 1) & (((unsigned int)a7 >> 9) ^ ((unsigned int)a7 >> 4));
  v27 = (__int64 *)(v25 + 56LL * v26);
  v28 = *v27;
  if ( a7 != *v27 )
  {
    v52 = 1;
    v53 = 0;
    while ( v28 != -8 )
    {
      if ( !v53 && v28 == -16 )
        v53 = v27;
      v26 = (v23 - 1) & (v52 + v26);
      v27 = (__int64 *)(v25 + 56LL * v26);
      v28 = *v27;
      if ( a7 == *v27 )
        goto LABEL_15;
      ++v52;
    }
    v54 = a8;
    if ( v53 )
      v27 = v53;
    ++*(_QWORD *)a8;
    v55 = *(_DWORD *)(v54 + 16) + 1;
    if ( 4 * v55 < 3 * v23 )
    {
      if ( v23 - *(_DWORD *)(a8 + 20) - v55 > v23 >> 3 )
      {
LABEL_85:
        v56 = a8;
        *(_DWORD *)(a8 + 16) = v55;
        if ( *v27 != -8 )
          --*(_DWORD *)(v56 + 20);
        v30 = (__int64)(v27 + 2);
        *v27 = v24;
        *((_DWORD *)v27 + 4) = 0;
        v27[3] = 0;
        v27[4] = (__int64)(v27 + 2);
        v27[5] = (__int64)(v27 + 2);
        v27[6] = 0;
        v19 = *a6;
        v65 = *a6 + a1[1];
        goto LABEL_88;
      }
LABEL_92:
      sub_18F57C0(a8, v23);
      sub_18F5560(a8, &a7, v70);
      v27 = (__int64 *)v70[0];
      v24 = a7;
      v55 = *(_DWORD *)(a8 + 16) + 1;
      goto LABEL_85;
    }
LABEL_91:
    v23 *= 2;
    goto LABEL_92;
  }
LABEL_15:
  v29 = (_QWORD *)v27[3];
  v30 = (__int64)(v27 + 2);
  if ( !v29 )
  {
LABEL_88:
    v47 = v30;
    goto LABEL_67;
  }
  v31 = v27 + 2;
  v32 = (_QWORD *)v27[3];
  do
  {
    if ( v18 > v32[4] )
    {
      v32 = (_QWORD *)v32[3];
    }
    else
    {
      v31 = v32;
      v32 = (_QWORD *)v32[2];
    }
  }
  while ( v32 );
  if ( (_QWORD *)v30 == v31 || (v38 = v31[5], v65 < v38) )
  {
    v47 = (__int64)(v27 + 2);
  }
  else
  {
    v58 = v31;
    if ( v18 <= v38 )
      v38 = *a6;
    v39 = v31[4];
    v19 = v38;
    if ( v65 >= v39 )
      v39 = v65;
    v65 = v39;
    v61 = sub_220EEE0(v31);
    v40 = sub_220F330(v58, v27 + 2);
    j_j___libc_free_0(v40, 48);
    v41 = v61;
    --v27[6];
    if ( v61 != v30 )
    {
      v62 = v19;
      v42 = v65;
      v57 = v10;
      v43 = v27;
      for ( i = v41; *(_QWORD *)(i + 40) <= v42; i = v45 )
      {
        if ( v42 < *(_QWORD *)(i + 32) )
          v42 = *(_QWORD *)(i + 32);
        v45 = sub_220EEE0(i);
        v46 = sub_220F330(i, v30);
        j_j___libc_free_0(v46, 48);
        --v43[6];
        if ( v45 == v30 )
          break;
      }
      v65 = v42;
      v27 = v43;
      v19 = v62;
      v11 = a1;
      v10 = v57;
    }
    v29 = (_QWORD *)v27[3];
    v47 = v30;
    if ( !v29 )
      goto LABEL_67;
  }
  do
  {
    if ( v29[4] < v65 )
    {
      v29 = (_QWORD *)v29[3];
    }
    else
    {
      v30 = (__int64)v29;
      v29 = (_QWORD *)v29[2];
    }
  }
  while ( v29 );
  if ( v30 == v47 || *(_QWORD *)(v30 + 32) > v65 )
  {
LABEL_67:
    v63 = (_QWORD *)v30;
    v59 = v47;
    v30 = sub_22077B0(48);
    *(_QWORD *)(v30 + 40) = 0;
    *(_QWORD *)(v30 + 32) = v65;
    v48 = sub_18F56C0(v27 + 1, v63, (__int64 *)(v30 + 32));
    if ( v49 )
    {
      v50 = v59 == v49 || v48 || *(_QWORD *)(v49 + 32) > v65;
      sub_220F040(v50, v30, v49, v59);
      ++v27[6];
    }
    else
    {
      v66 = v48;
      j_j___libc_free_0(v30, 48);
      v30 = (__int64)v66;
    }
  }
  *(_QWORD *)(v30 + 40) = v19;
  v51 = v27[4];
  v17 = *a5;
  if ( *(_QWORD *)(v51 + 40) <= *a5 && *(_QWORD *)(v51 + 32) >= v17 + v10[1] )
    return 1;
  if ( byte_4FAE400 )
  {
    v18 = *a6;
    if ( v17 <= *a6 )
    {
LABEL_41:
      v22 = v10[1];
LABEL_36:
      if ( (__int64)(v17 + v22) > v18 )
      {
        v35 = v11[1] + v18 - v17;
        result = 3;
        if ( v22 >= v35 )
          return result;
      }
    }
  }
  if ( byte_4FAE4E0 )
    return 4;
  v18 = *a6;
LABEL_22:
  if ( v18 <= v17 )
  {
    v33 = v11[1] + v18;
    result = 0;
    if ( v33 > v17 )
      return result;
    return 4;
  }
  v36 = v10[1] + v17;
  if ( v36 <= v18 )
    return 4;
  v37 = v11[1] + v18;
  result = 2;
  if ( v36 > v37 )
    return 4;
  return result;
}
