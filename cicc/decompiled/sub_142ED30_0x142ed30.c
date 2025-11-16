// Function: sub_142ED30
// Address: 0x142ed30
//
_QWORD *__fastcall sub_142ED30(__int64 a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v4; // rax
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rbx
  bool v7; // zf
  _QWORD *v8; // rax
  _QWORD *v9; // rbx
  _QWORD *v10; // r15
  _QWORD *v11; // r13
  _QWORD *result; // rax
  unsigned __int64 v13; // rbx
  __int64 *v14; // r15
  __int64 v15; // r14
  _QWORD *v16; // r8
  _QWORD *v17; // rdi
  _QWORD *v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rcx
  _QWORD *v21; // r9
  __int64 v22; // rcx
  __int64 v23; // rdx
  _QWORD *v24; // r13
  __int64 v25; // rax
  _QWORD *v26; // rax
  _QWORD *v27; // rdx
  _QWORD *v28; // r13
  _BOOL8 v29; // rdi
  char *v30; // rsi
  _QWORD *v31; // rdi
  _QWORD *v32; // r9
  _QWORD *v33; // rdx
  __int64 v34; // rsi
  __int64 v35; // rcx
  _QWORD *v36; // r14
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // rax
  _QWORD *v40; // rax
  _QWORD *v41; // rdx
  _BOOL8 v42; // rdi
  __int64 v43; // rax
  _QWORD *v44; // rax
  __int64 v45; // rdx
  _BOOL8 v46; // rdi
  _QWORD *v47; // r10
  _QWORD *v48; // rdi
  _QWORD *v49; // [rsp+8h] [rbp-128h]
  _QWORD *v50; // [rsp+8h] [rbp-128h]
  _QWORD *v51; // [rsp+10h] [rbp-120h]
  _QWORD *v52; // [rsp+10h] [rbp-120h]
  __int64 v53; // [rsp+10h] [rbp-120h]
  _QWORD *v54; // [rsp+10h] [rbp-120h]
  unsigned __int64 v55; // [rsp+18h] [rbp-118h]
  unsigned __int64 v56; // [rsp+18h] [rbp-118h]
  __int64 v57; // [rsp+18h] [rbp-118h]
  _QWORD *v58; // [rsp+18h] [rbp-118h]
  __int64 v59; // [rsp+18h] [rbp-118h]
  _QWORD *v60; // [rsp+28h] [rbp-108h] BYREF
  unsigned __int64 v61; // [rsp+30h] [rbp-100h] BYREF
  unsigned __int64 v62[2]; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v63; // [rsp+50h] [rbp-E0h] BYREF
  __m128i v64; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD *v65; // [rsp+70h] [rbp-C0h]
  _QWORD *v66; // [rsp+78h] [rbp-B8h]
  __int64 v67; // [rsp+80h] [rbp-B0h]

  v4 = (_QWORD *)*a3;
  *a3 = 0;
  v60 = v4;
  sub_15E4EB0(v62);
  v5 = v62[1];
  v55 = v62[0];
  sub_16C1840(&v64);
  sub_16C1A90(&v64, v55, v5);
  sub_16C1AA0(&v64, &v61);
  v6 = v61;
  if ( (__int64 *)v62[0] != &v63 )
    j_j___libc_free_0(v62[0], v63 + 1);
  v7 = *(_BYTE *)(a1 + 178) == 0;
  v62[0] = v6;
  if ( v7 )
  {
    v64.m128i_i64[1] = 0;
    v64.m128i_i64[0] = (__int64)byte_3F871B3;
  }
  else
  {
    v64.m128i_i64[0] = 0;
  }
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v8 = sub_142DA40((_QWORD *)a1, v62, &v64);
  v9 = v66;
  v10 = v65;
  v11 = v8;
  v56 = (unsigned __int64)(v8 + 4);
  if ( v66 != v65 )
  {
    do
    {
      if ( *v10 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v10 + 8LL))(*v10);
      ++v10;
    }
    while ( v9 != v10 );
    v10 = v65;
  }
  if ( v10 )
    j_j___libc_free_0(v10, v67 - (_QWORD)v10);
  v11[5] = a2;
  result = v60;
  v13 = v60[2];
  v14 = (__int64 *)((unsigned __int16)(4 * *(unsigned __int8 *)(a1 + 178)) & 0xFFF8 | v56 & 0xFFFFFFFFFFFFFFF8LL);
  v15 = *v14;
  if ( v13 != *v14 && v13 )
  {
    result = *(_QWORD **)(a1 + 144);
    v16 = (_QWORD *)(a1 + 136);
    if ( !result )
    {
      v21 = (_QWORD *)(a1 + 136);
LABEL_27:
      v24 = (_QWORD *)(a1 + 128);
      goto LABEL_28;
    }
    v17 = (_QWORD *)(a1 + 136);
    v18 = *(_QWORD **)(a1 + 144);
    do
    {
      while ( 1 )
      {
        v19 = v18[2];
        v20 = v18[3];
        if ( v13 <= v18[4] )
          break;
        v18 = (_QWORD *)v18[3];
        if ( !v20 )
          goto LABEL_19;
      }
      v17 = v18;
      v18 = (_QWORD *)v18[2];
    }
    while ( v19 );
LABEL_19:
    if ( v16 == v17 || v13 < v17[4] )
    {
LABEL_21:
      v21 = v16;
      do
      {
        while ( 1 )
        {
          v22 = result[2];
          v23 = result[3];
          if ( v13 <= result[4] )
            break;
          result = (_QWORD *)result[3];
          if ( !v23 )
            goto LABEL_25;
        }
        v21 = result;
        result = (_QWORD *)result[2];
      }
      while ( v22 );
LABEL_25:
      if ( v16 != v21 && v13 >= v21[4] )
        goto LABEL_33;
      goto LABEL_27;
    }
    v32 = (_QWORD *)(a1 + 136);
    v33 = *(_QWORD **)(a1 + 144);
    do
    {
      while ( 1 )
      {
        v34 = v33[2];
        v35 = v33[3];
        if ( v13 <= v33[4] )
          break;
        v33 = (_QWORD *)v33[3];
        if ( !v35 )
          goto LABEL_46;
      }
      v32 = v33;
      v33 = (_QWORD *)v33[2];
    }
    while ( v34 );
LABEL_46:
    if ( v16 == v32 || v13 < v32[4] )
    {
      v53 = a1 + 136;
      v24 = (_QWORD *)(a1 + 128);
      v50 = v32;
      v43 = sub_22077B0(48);
      *(_QWORD *)(v43 + 32) = v13;
      *(_QWORD *)(v43 + 40) = 0;
      v59 = v43;
      v44 = sub_142EC30((_QWORD *)(a1 + 128), v50, (unsigned __int64 *)(v43 + 32));
      if ( v45 )
      {
        v46 = v53 == v45 || v44 || v13 < *(_QWORD *)(v45 + 32);
        sub_220F040(v46, v59, v45, v53);
        v47 = (_QWORD *)v59;
        ++*(_QWORD *)(a1 + 168);
        v16 = (_QWORD *)(a1 + 136);
      }
      else
      {
        v54 = v44;
        j_j___libc_free_0(v59, 48);
        v16 = (_QWORD *)(a1 + 136);
        v47 = v54;
      }
      result = *(_QWORD **)(a1 + 144);
      if ( v15 == v47[5] )
      {
        if ( !result )
        {
          v21 = v16;
LABEL_28:
          v51 = v21;
          v49 = v16;
          v25 = sub_22077B0(48);
          *(_QWORD *)(v25 + 32) = v13;
          *(_QWORD *)(v25 + 40) = 0;
          v57 = v25;
          v26 = sub_142EC30(v24, v51, (unsigned __int64 *)(v25 + 32));
          v28 = v26;
          if ( v27 )
          {
            v29 = v49 == v27 || v26 || v13 < v27[4];
            result = (_QWORD *)sub_220F040(v29, v57, v27, v49);
            v21 = (_QWORD *)v57;
            ++*(_QWORD *)(a1 + 168);
          }
          else
          {
            result = (_QWORD *)j_j___libc_free_0(v57, 48);
            v21 = v28;
          }
LABEL_33:
          v21[5] = v15;
          goto LABEL_34;
        }
        goto LABEL_21;
      }
      if ( !result )
      {
        v36 = v16;
        goto LABEL_56;
      }
    }
    else if ( v15 == v32[5] )
    {
      goto LABEL_21;
    }
    v36 = v16;
    do
    {
      while ( 1 )
      {
        v37 = result[2];
        v38 = result[3];
        if ( v13 <= result[4] )
          break;
        result = (_QWORD *)result[3];
        if ( !v38 )
          goto LABEL_53;
      }
      v36 = result;
      result = (_QWORD *)result[2];
    }
    while ( v37 );
LABEL_53:
    if ( v16 != v36 && v13 >= v36[4] )
      goto LABEL_73;
    v24 = (_QWORD *)(a1 + 128);
LABEL_56:
    v52 = v16;
    v58 = v36;
    v39 = sub_22077B0(48);
    *(_QWORD *)(v39 + 32) = v13;
    v36 = (_QWORD *)v39;
    *(_QWORD *)(v39 + 40) = 0;
    v40 = sub_142EC30(v24, v58, (unsigned __int64 *)(v39 + 32));
    if ( v41 )
    {
      v42 = v52 == v41 || v40 || v13 < v41[4];
      result = (_QWORD *)sub_220F040(v42, v36, v41, v52);
      ++*(_QWORD *)(a1 + 168);
    }
    else
    {
      v48 = v36;
      v36 = v40;
      result = (_QWORD *)j_j___libc_free_0(v48, 48);
    }
LABEL_73:
    v36[5] = 0;
  }
LABEL_34:
  v30 = (char *)v14[4];
  if ( v30 == (char *)v14[5] )
  {
    result = (_QWORD *)sub_142DF10(v14 + 3, v30, &v60);
    v31 = v60;
  }
  else
  {
    if ( v30 )
    {
      result = v60;
      *(_QWORD *)v30 = v60;
      v14[4] += 8;
      return result;
    }
    v14[4] = 8;
    v31 = v60;
  }
  if ( v31 )
    return (_QWORD *)(*(__int64 (__fastcall **)(_QWORD *))(*v31 + 8LL))(v31);
  return result;
}
