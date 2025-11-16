// Function: sub_D79860
// Address: 0xd79860
//
__int64 __fastcall sub_D79860(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 result; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // r14
  __int8 v12; // al
  unsigned __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rsi
  unsigned __int8 *v16; // rax
  __int64 v17; // r12
  __int64 v18; // r15
  char v19; // bl
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r12
  __int64 v24; // r13
  __int64 v25; // r12
  unsigned __int8 *i; // rcx
  __int64 v27; // rdi
  unsigned __int8 *v28; // r12
  __int64 v29; // rdx
  __m128i *v30; // rsi
  __int64 v31; // rbx
  int v32; // ebx
  char v33; // bl
  __m128i v34; // rax
  __int64 v35; // rax
  int v36; // ebx
  __int64 v37; // [rsp+8h] [rbp-B8h]
  __int64 v38; // [rsp+8h] [rbp-B8h]
  unsigned int v43; // [rsp+30h] [rbp-90h]
  __int64 v44; // [rsp+30h] [rbp-90h]
  __int64 v45; // [rsp+30h] [rbp-90h]
  __int64 v47; // [rsp+40h] [rbp-80h] BYREF
  __int64 v48; // [rsp+48h] [rbp-78h] BYREF
  __int64 v49; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v50; // [rsp+58h] [rbp-68h]
  char v51; // [rsp+5Ch] [rbp-64h]
  __int64 v52; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v53; // [rsp+68h] [rbp-58h]
  char v54; // [rsp+6Ch] [rbp-54h]
  unsigned __int64 v55; // [rsp+70h] [rbp-50h] BYREF
  __int8 v56; // [rsp+78h] [rbp-48h]
  __m128i v57; // [rsp+80h] [rbp-40h] BYREF

  if ( *(_BYTE *)(*((_QWORD *)a1 + 1) + 8LL) == 14 )
  {
    v28 = sub_BD3990(a1, a2);
    if ( *v28 == 1 )
    {
      if ( **((_BYTE **)v28 - 4) )
        goto LABEL_2;
    }
    else if ( *v28 )
    {
      goto LABEL_2;
    }
    result = (__int64)sub_BD5D20((__int64)v28);
    if ( v29 != 18
      || *(_QWORD *)result ^ 0x75705F6178635F5FLL | *(_QWORD *)(result + 8) ^ 0x75747269765F6572LL
      || *(_WORD *)(result + 16) != 27745 )
    {
      v57.m128i_i64[0] = sub_D789D0(a4, (__int64)v28);
      v57.m128i_i64[1] = a2;
      v30 = *(__m128i **)(a5 + 8);
      if ( v30 == *(__m128i **)(a5 + 16) )
      {
        return sub_9D2FF0((const __m128i **)a5, v30, &v57);
      }
      else
      {
        if ( v30 )
        {
          *v30 = _mm_loadu_si128(&v57);
          v30 = *(__m128i **)(a5 + 8);
        }
        *(_QWORD *)(a5 + 8) = v30 + 1;
        return a5;
      }
    }
    return result;
  }
LABEL_2:
  v6 = a3 + 312;
  result = *a1;
  if ( (_BYTE)result == 10 )
  {
    v8 = *((_QWORD *)a1 + 1);
    v9 = sub_AE4AC0(a3 + 312, v8);
    v10 = *(_QWORD *)(v8 + 16);
    v11 = v9;
    result = v10 + 8LL * *(unsigned int *)(v8 + 12);
    v37 = result;
    if ( v10 != result )
    {
      v43 = 0;
      do
      {
        v12 = *(_BYTE *)(v11 + 16LL * v43 + 32);
        v57.m128i_i64[0] = *(_QWORD *)(v11 + 16LL * v43 + 24);
        v57.m128i_i8[8] = v12;
        v13 = sub_CA1930(&v57);
        v14 = (unsigned int)sub_AE1C80(v11, v13);
        v15 = sub_CA1930(&v57) + a2;
        if ( (a1[7] & 0x40) != 0 )
          v16 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
        else
          v16 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
        v10 += 8;
        result = sub_D79860(*(_QWORD *)&v16[32 * v14], v15, a3, a4, a5, a6);
        ++v43;
      }
      while ( v37 != v10 );
    }
  }
  else if ( (_BYTE)result == 9 )
  {
    v17 = *((_QWORD *)a1 + 1);
    v18 = *(_QWORD *)(v17 + 24);
    v19 = sub_AE5020(a3 + 312, v18);
    v20 = sub_9208B0(v6, v18);
    v57.m128i_i64[1] = v21;
    v57.m128i_i64[0] = ((1LL << v19) + ((unsigned __int64)(v20 + 7) >> 3) - 1) >> v19 << v19;
    result = sub_CA1930(&v57);
    v22 = *(_QWORD *)(v17 + 32);
    if ( (_DWORD)v22 )
    {
      v44 = result;
      v24 = a2;
      v38 = 32LL * (unsigned int)v22;
      v25 = 0;
      if ( (a1[7] & 0x40) == 0 )
        goto LABEL_16;
LABEL_13:
      for ( i = (unsigned __int8 *)*((_QWORD *)a1 - 1); ; i = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)] )
      {
        v27 = *(_QWORD *)&i[v25];
        v25 += 32;
        result = sub_D79860(v27, v24, a3, a4, a5, a6);
        v24 += v44;
        if ( v25 == v38 )
          break;
        if ( (a1[7] & 0x40) != 0 )
          goto LABEL_13;
LABEL_16:
        ;
      }
    }
  }
  else if ( (_BYTE)result == 5 && *((_WORD *)a1 + 1) == 38 )
  {
    result = -32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
    v31 = *(_QWORD *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    if ( *(_BYTE *)v31 == 5 && *(_WORD *)(v31 + 2) == 15 )
    {
      v51 = 0;
      v50 = 1;
      v49 = 0;
      v53 = 1;
      v52 = 0;
      v54 = 0;
      result = sub_96E080(
                 *(_QWORD *)(v31 - 32LL * (*(_DWORD *)(v31 + 4) & 0x7FFFFFF)),
                 &v47,
                 (__int64)&v49,
                 a3 + 312,
                 0);
      if ( (_BYTE)result )
      {
        result = sub_96E080(
                   *(_QWORD *)(v31 + 32 * (1LL - (*(_DWORD *)(v31 + 4) & 0x7FFFFFF))),
                   &v48,
                   (__int64)&v52,
                   v6,
                   0);
        if ( (_BYTE)result )
        {
          result = a6;
          if ( a6 == v48 )
          {
            v57.m128i_i8[12] = 0;
            v57.m128i_i32[2] = 64;
            v57.m128i_i64[0] = 0;
            result = sub_AA8A40(&v49, v57.m128i_i64);
            v32 = result;
            if ( v57.m128i_i32[2] > 0x40u && v57.m128i_i64[0] )
              result = j_j___libc_free_0_0(v57.m128i_i64[0]);
            if ( !v32 )
            {
              v45 = *(_QWORD *)(*(_QWORD *)(a6 - 32) + 8LL);
              v33 = sub_AE5020(v6, v45);
              v34.m128i_i64[0] = sub_9208B0(v6, v45);
              v57 = v34;
              v55 = ((1LL << v33) + ((unsigned __int64)(v34.m128i_i64[0] + 7) >> 3) - 1) >> v33 << v33;
              v56 = v34.m128i_i8[8];
              v35 = sub_CA1930(&v55);
              v57.m128i_i8[12] = 0;
              v57.m128i_i32[2] = 64;
              v57.m128i_i64[0] = v35;
              result = sub_AA8A40(&v52, v57.m128i_i64);
              v36 = result;
              if ( v57.m128i_i32[2] > 0x40u && v57.m128i_i64[0] )
                result = j_j___libc_free_0_0(v57.m128i_i64[0]);
              if ( v36 <= 0 )
                result = sub_D79860(v47, a2, a3, a4, a5, a6);
            }
          }
        }
      }
      if ( v53 > 0x40 && v52 )
        result = j_j___libc_free_0_0(v52);
      if ( v50 > 0x40 && v49 )
        return j_j___libc_free_0_0(v49);
    }
  }
  return result;
}
