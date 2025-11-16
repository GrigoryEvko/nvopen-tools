// Function: sub_1D61F00
// Address: 0x1d61f00
//
__int64 __fastcall sub_1D61F00(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // rcx
  __int64 v4; // r13
  __int64 v6; // rdi
  __int64 v7; // rax
  int v8; // edx
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rax
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // rax
  __m128i *v15; // rax
  unsigned int v17; // r14d
  __int64 v18; // rax
  __int64 v19; // rcx
  int v20; // r8d
  int v21; // r9d
  __m128i *v22; // rax
  __int64 v23; // rdx
  char v24; // [rsp+7h] [rbp-69h] BYREF
  __int64 v25; // [rsp+8h] [rbp-68h] BYREF
  __m128i v26; // [rsp+10h] [rbp-60h] BYREF
  __m128i v27; // [rsp+20h] [rbp-50h] BYREF
  __m128i v28; // [rsp+30h] [rbp-40h] BYREF
  __int64 v29; // [rsp+40h] [rbp-30h]

  v3 = a3;
  v4 = 0;
  v6 = *(_QWORD *)(a1 + 80);
  v7 = *(unsigned int *)(v6 + 8);
  if ( (_DWORD)v7 )
    v4 = *(_QWORD *)(*(_QWORD *)v6 + 8 * v7 - 8);
  v8 = *(unsigned __int8 *)(a2 + 16);
  if ( (_BYTE)v8 == 13 )
  {
    v9 = *(_DWORD *)(a2 + 32);
    v10 = *(__int64 **)(a2 + 24);
    if ( v9 <= 0x40 )
      v11 = (__int64)((_QWORD)v10 << (64 - (unsigned __int8)v9)) >> (64 - (unsigned __int8)v9);
    else
      v11 = *v10;
    *(_QWORD *)(*(_QWORD *)(a1 + 56) + 8LL) += v11;
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 8)
                                                                                           + 736LL))(
           *(_QWORD *)(a1 + 8),
           *(_QWORD *)(a1 + 24),
           *(_QWORD *)(a1 + 56),
           *(_QWORD *)(a1 + 32),
           *(unsigned int *)(a1 + 40),
           0) )
    {
      return 1;
    }
    v12 = *(_DWORD *)(a2 + 32);
    v13 = *(__int64 **)(a2 + 24);
    if ( v12 <= 0x40 )
      v14 = (__int64)((_QWORD)v13 << (64 - (unsigned __int8)v12)) >> (64 - (unsigned __int8)v12);
    else
      v14 = *v13;
    *(_QWORD *)(*(_QWORD *)(a1 + 56) + 8LL) -= v14;
    v15 = *(__m128i **)(a1 + 56);
    goto LABEL_15;
  }
  if ( (unsigned __int8)v8 > 3u )
  {
    if ( (unsigned __int8)v8 <= 0x17u )
    {
      v25 = 0;
      if ( (_BYTE)v8 != 5 )
      {
        if ( (_BYTE)v8 == 15 )
          return 1;
        goto LABEL_33;
      }
      if ( (unsigned int)v3 > 4 )
      {
LABEL_32:
        sub_1D5ABA0((__int64 *)v6, v4);
LABEL_33:
        v15 = *(__m128i **)(a1 + 56);
        goto LABEL_15;
      }
      if ( (unsigned __int8)sub_1D62410(a1, a2, *(unsigned __int16 *)(a2 + 18), v3, 0) )
        return 1;
    }
    else
    {
      v15 = *(__m128i **)(a1 + 56);
      v25 = a2;
      v24 = 0;
      v26 = _mm_loadu_si128(v15);
      v27 = _mm_loadu_si128(v15 + 1);
      v28 = _mm_loadu_si128(v15 + 2);
      v29 = v15[3].m128i_i64[0];
      v17 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
      if ( (unsigned int)v3 > 4 )
        goto LABEL_15;
      if ( !(unsigned __int8)sub_1D62410(a1, a2, (unsigned int)(v8 - 24), v3, &v24) )
        goto LABEL_33;
      if ( v24 )
        return 1;
      v18 = *(_QWORD *)(v25 + 8);
      if ( v18 && !*(_QWORD *)(v18 + 8) || (unsigned __int8)sub_1D63010(a1, v25, &v26, *(_QWORD *)(a1 + 56)) )
      {
        sub_14EF3D0(*(_QWORD *)a1, &v25);
        return 1;
      }
      v22 = *(__m128i **)(a1 + 56);
      *v22 = _mm_loadu_si128(&v26);
      v22[1] = _mm_loadu_si128(&v27);
      v22[2] = _mm_loadu_si128(&v28);
      v23 = v29;
      v22[3].m128i_i64[0] = v29;
      sub_1D61E90(*(_QWORD *)a1, v17, v23, v19, v20, v21);
    }
    v6 = *(_QWORD *)(a1 + 80);
    goto LABEL_32;
  }
  v15 = *(__m128i **)(a1 + 56);
  if ( !v15->m128i_i64[0] )
  {
    v15->m128i_i64[0] = a2;
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 8)
                                                                                           + 736LL))(
           *(_QWORD *)(a1 + 8),
           *(_QWORD *)(a1 + 24),
           *(_QWORD *)(a1 + 56),
           *(_QWORD *)(a1 + 32),
           *(unsigned int *)(a1 + 40),
           0) )
    {
      return 1;
    }
    **(_QWORD **)(a1 + 56) = 0;
    v15 = *(__m128i **)(a1 + 56);
  }
LABEL_15:
  if ( !v15[1].m128i_i8[0] )
  {
    v15[1].m128i_i8[0] = 1;
    *(_QWORD *)(*(_QWORD *)(a1 + 56) + 32LL) = a2;
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 8)
                                                                                           + 736LL))(
           *(_QWORD *)(a1 + 8),
           *(_QWORD *)(a1 + 24),
           *(_QWORD *)(a1 + 56),
           *(_QWORD *)(a1 + 32),
           *(unsigned int *)(a1 + 40),
           0) )
    {
      return 1;
    }
    *(_BYTE *)(*(_QWORD *)(a1 + 56) + 16LL) = 0;
    *(_QWORD *)(*(_QWORD *)(a1 + 56) + 32LL) = 0;
    v15 = *(__m128i **)(a1 + 56);
  }
  if ( !v15[1].m128i_i64[1] )
  {
    v15[1].m128i_i64[1] = 1;
    *(_QWORD *)(*(_QWORD *)(a1 + 56) + 40LL) = a2;
    if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 8)
                                                                                            + 736LL))(
            *(_QWORD *)(a1 + 8),
            *(_QWORD *)(a1 + 24),
            *(_QWORD *)(a1 + 56),
            *(_QWORD *)(a1 + 32),
            *(unsigned int *)(a1 + 40),
            0) )
    {
      *(_QWORD *)(*(_QWORD *)(a1 + 56) + 24LL) = 0;
      *(_QWORD *)(*(_QWORD *)(a1 + 56) + 40LL) = 0;
      goto LABEL_21;
    }
    return 1;
  }
LABEL_21:
  sub_1D5ABA0(*(__int64 **)(a1 + 80), v4);
  return 0;
}
