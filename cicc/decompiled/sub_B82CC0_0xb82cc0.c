// Function: sub_B82CC0
// Address: 0xb82cc0
//
_BYTE *__fastcall sub_B82CC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, __int64 a6, __int64 a7)
{
  _BYTE *result; // rax
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rax
  _BYTE *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // r14
  __int64 v18; // r14
  __int64 v19; // rax
  __m128i *v20; // rax
  __int64 v21; // rdx
  __m128i *v22; // rsi
  unsigned __int64 v23; // rdx
  __int8 *v24; // r13
  size_t v25; // r12
  __int64 v26; // rax
  __m128i *v27; // rdx
  size_t v28; // rdx
  _QWORD *v29; // rbx
  _QWORD *v30; // r12
  _QWORD *v31; // rdi
  __int64 v32; // rbx
  _BYTE *v33; // r15
  __int64 v34; // rsi
  __m128i **v35; // r12
  __int64 v36; // rdx
  __int64 v37; // r13
  __m128i v38; // xmm1
  __m128i *v39; // rdx
  __m128i **v40; // rax
  __m128i *v41; // rdi
  __int64 v42; // [rsp+10h] [rbp-360h]
  __int64 v43; // [rsp+18h] [rbp-358h]
  __int64 v44; // [rsp+20h] [rbp-350h]
  __int64 v48; // [rsp+48h] [rbp-328h] BYREF
  _QWORD v49[2]; // [rsp+50h] [rbp-320h] BYREF
  __int64 v50; // [rsp+60h] [rbp-310h] BYREF
  __int64 *v51; // [rsp+70h] [rbp-300h]
  __int64 v52; // [rsp+80h] [rbp-2F0h] BYREF
  __int64 v53[2]; // [rsp+A0h] [rbp-2D0h] BYREF
  _QWORD v54[2]; // [rsp+B0h] [rbp-2C0h] BYREF
  __int64 *v55; // [rsp+C0h] [rbp-2B0h]
  __int64 v56; // [rsp+D0h] [rbp-2A0h] BYREF
  __m128i v57; // [rsp+F0h] [rbp-280h] BYREF
  __m128i v58; // [rsp+100h] [rbp-270h] BYREF
  __int64 *v59; // [rsp+110h] [rbp-260h]
  __int64 v60; // [rsp+120h] [rbp-250h] BYREF
  __m128i v61; // [rsp+140h] [rbp-230h] BYREF
  __m128i v62; // [rsp+150h] [rbp-220h] BYREF
  __int64 *v63; // [rsp+160h] [rbp-210h]
  __int64 v64; // [rsp+170h] [rbp-200h] BYREF
  __int64 v65[10]; // [rsp+190h] [rbp-1E0h] BYREF
  _QWORD *v66; // [rsp+1E0h] [rbp-190h]
  unsigned int v67; // [rsp+1E8h] [rbp-188h]
  _BYTE v68[384]; // [rsp+1F0h] [rbp-180h] BYREF

  result = (_BYTE *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 120LL))(a2);
  if ( result )
    return result;
  v48 = a6;
  v44 = a7;
  if ( a7 )
  {
    sub_B7F0C0(&v48, a7);
    v11 = a7;
LABEL_5:
    v12 = *(_QWORD *)(v11 + 80);
    v61 = 0u;
    v42 = a5 + a4;
    v13 = v12 - 24;
    if ( !v12 )
      v13 = 0;
    v43 = v13;
    sub_B17850((__int64)v65, (__int64)"size-info", (__int64)"IRSizeChange", 12, &v61, v13);
    v14 = (_BYTE *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2);
    sub_B16430((__int64)v49, "Pass", 4u, v14, v15);
    v16 = sub_B826F0((__int64)v65, (__int64)v49);
    sub_B18290(v16, ": IR instruction count changed from ", 0x24u);
    sub_B169E0(v53, "IRInstrsBefore", 14, a5);
    v17 = sub_B826F0(v16, (__int64)v53);
    sub_B18290(v17, " to ", 4u);
    sub_B167F0(v57.m128i_i64, "IRInstrsAfter", 13, v42);
    v18 = sub_B826F0(v17, (__int64)&v57);
    sub_B18290(v18, "; Delta: ", 9u);
    sub_B167F0(v61.m128i_i64, "DeltaInstrCount", 15, a4);
    sub_B826F0(v18, (__int64)&v61);
    if ( v63 != &v64 )
      j_j___libc_free_0(v63, v64 + 1);
    if ( (__m128i *)v61.m128i_i64[0] != &v62 )
      j_j___libc_free_0(v61.m128i_i64[0], v62.m128i_i64[0] + 1);
    if ( v59 != &v60 )
      j_j___libc_free_0(v59, v60 + 1);
    if ( (__m128i *)v57.m128i_i64[0] != &v58 )
      j_j___libc_free_0(v57.m128i_i64[0], v58.m128i_i64[0] + 1);
    if ( v55 != &v56 )
      j_j___libc_free_0(v55, v56 + 1);
    if ( (_QWORD *)v53[0] != v54 )
      j_j___libc_free_0(v53[0], v54[0] + 1LL);
    if ( v51 != &v52 )
      j_j___libc_free_0(v51, v52 + 1);
    if ( (__int64 *)v49[0] != &v50 )
      j_j___libc_free_0(v49[0], v50 + 1);
    v19 = sub_B2BE50(a7);
    sub_B6EB20(v19, (__int64)v65);
    v20 = (__m128i *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2);
    v53[0] = (__int64)v54;
    v22 = v20;
    if ( v20 )
    {
      sub_B7EB70(v53, v20, (__int64)v20->m128i_i64 + v21);
    }
    else
    {
      v53[1] = 0;
      LOBYTE(v54[0]) = 0;
    }
    v57.m128i_i64[0] = a6;
    v57.m128i_i64[1] = (__int64)&a7;
    v58.m128i_i64[0] = v43;
    v58.m128i_i64[1] = (__int64)v53;
    if ( v44 )
    {
      v24 = (__int8 *)sub_BD5D20(a7);
      v25 = v23;
      if ( !v24 )
      {
        v62.m128i_i8[0] = 0;
        v28 = 0;
        v61 = (__m128i)(unsigned __int64)&v62;
        v22 = &v62;
LABEL_31:
        sub_B827A0(v57.m128i_i64, v22, v28);
        if ( (__m128i *)v61.m128i_i64[0] != &v62 )
        {
          v22 = (__m128i *)(v62.m128i_i64[0] + 1);
          j_j___libc_free_0(v61.m128i_i64[0], v62.m128i_i64[0] + 1);
        }
LABEL_33:
        if ( (_QWORD *)v53[0] != v54 )
        {
          v22 = (__m128i *)(v54[0] + 1LL);
          j_j___libc_free_0(v53[0], v54[0] + 1LL);
        }
        v29 = v66;
        v65[0] = (__int64)&unk_49D9D40;
        v30 = &v66[10 * v67];
        if ( v66 != v30 )
        {
          do
          {
            v30 -= 10;
            v31 = (_QWORD *)v30[4];
            if ( v31 != v30 + 6 )
            {
              v22 = (__m128i *)(v30[6] + 1LL);
              j_j___libc_free_0(v31, v22);
            }
            if ( (_QWORD *)*v30 != v30 + 2 )
            {
              v22 = (__m128i *)(v30[2] + 1LL);
              j_j___libc_free_0(*v30, v22);
            }
          }
          while ( v29 != v30 );
          v30 = v66;
        }
        result = v68;
        if ( v30 != (_QWORD *)v68 )
          return (_BYTE *)_libc_free(v30, v22);
        return result;
      }
      v49[0] = v23;
      v26 = v23;
      v61.m128i_i64[0] = (__int64)&v62;
      if ( v23 > 0xF )
      {
        v61.m128i_i64[0] = sub_22409D0(&v61, v49, 0);
        v41 = (__m128i *)v61.m128i_i64[0];
        v62.m128i_i64[0] = v49[0];
      }
      else
      {
        if ( v23 == 1 )
        {
          v62.m128i_i8[0] = *v24;
          v27 = &v62;
LABEL_30:
          v61.m128i_i64[1] = v26;
          v27->m128i_i8[v26] = 0;
          v28 = v61.m128i_u64[1];
          v22 = (__m128i *)v61.m128i_i64[0];
          goto LABEL_31;
        }
        if ( !v23 )
        {
          v27 = &v62;
          goto LABEL_30;
        }
        v41 = &v62;
      }
      memcpy(v41, v24, v25);
      v26 = v49[0];
      v27 = (__m128i *)v61.m128i_i64[0];
      goto LABEL_30;
    }
    v35 = *(__m128i ***)a6;
    v36 = *(unsigned int *)(a6 + 8);
    v37 = *(_QWORD *)a6 + 8 * v36;
    if ( (_DWORD)v36 )
    {
      while ( *v35 == (__m128i *)-8LL || !*v35 )
        ++v35;
    }
    v38 = _mm_loadu_si128(&v58);
    v61 = _mm_loadu_si128(&v57);
    v62 = v38;
    if ( (__m128i **)v37 == v35 )
      goto LABEL_33;
    while ( 1 )
    {
      v22 = *v35 + 1;
      sub_B827A0(v61.m128i_i64, v22, (*v35)->m128i_i64[0]);
      v39 = v35[1];
      v40 = v35 + 1;
      if ( v39 )
        goto LABEL_60;
      do
      {
        do
        {
          v39 = v40[1];
          ++v40;
        }
        while ( !v39 );
LABEL_60:
        ;
      }
      while ( v39 == (__m128i *)-8LL );
      if ( (__m128i **)v37 == v40 )
        goto LABEL_33;
      v35 = v40;
    }
  }
  v32 = *(_QWORD *)(a3 + 32);
  v33 = (_BYTE *)(a3 + 24);
  result = (_BYTE *)a6;
  v65[0] = a6;
  if ( a3 + 24 != v32 )
  {
    do
    {
      v34 = v32 - 56;
      if ( !v32 )
        v34 = 0;
      sub_B7F0C0(v65, v34);
      v32 = *(_QWORD *)(v32 + 8);
    }
    while ( v33 != (_BYTE *)v32 );
    result = *(_BYTE **)(a3 + 32);
    if ( v33 != result )
    {
      while ( 1 )
      {
        if ( !result )
          BUG();
        if ( result + 16 != (_BYTE *)(*((_QWORD *)result + 2) & 0xFFFFFFFFFFFFFFF8LL) )
          break;
        result = (_BYTE *)*((_QWORD *)result + 1);
        if ( v33 == result )
          return result;
      }
      v11 = (__int64)(result - 56);
      a7 = v11;
      goto LABEL_5;
    }
  }
  return result;
}
