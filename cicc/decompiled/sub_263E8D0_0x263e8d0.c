// Function: sub_263E8D0
// Address: 0x263e8d0
//
__m128i *__fastcall sub_263E8D0(_QWORD *a1, __int64 a2, const __m128i **a3)
{
  __int64 v6; // rax
  _QWORD *v7; // rcx
  __m128i *v8; // r12
  const __m128i *v9; // rax
  unsigned __int64 *v10; // rsi
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rax
  bool v15; // al
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  char v21; // di
  unsigned __int64 v22; // rax
  unsigned __int32 v23; // edi
  unsigned __int32 v24; // eax
  unsigned __int64 v25; // [rsp+18h] [rbp-38h]
  _QWORD *v26; // [rsp+18h] [rbp-38h]

  v6 = sub_22077B0(0x38u);
  v7 = a1 + 1;
  v8 = (__m128i *)v6;
  v9 = *a3;
  v8[3].m128i_i64[0] = 0;
  v10 = (unsigned __int64 *)&v8[2];
  v8[2] = _mm_loadu_si128(v9);
  if ( a1 + 1 != (_QWORD *)a2 )
  {
    v11 = v8[2].m128i_u64[0];
    v12 = *(_QWORD *)(a2 + 32);
    v13 = a2;
    if ( v11 < v12 )
    {
LABEL_3:
      if ( a1[3] != a2 )
      {
        v14 = sub_220EF80(a2);
        v7 = a1 + 1;
        if ( v11 <= *(_QWORD *)(v14 + 32) )
        {
          v10 = (unsigned __int64 *)&v8[2];
          if ( v11 != *(_QWORD *)(v14 + 32) || *(_DWORD *)(v14 + 40) >= v8[2].m128i_i32[2] )
          {
LABEL_13:
            v26 = v7;
            v18 = sub_263E3C0((__int64)a1, v10);
            v7 = v26;
            a2 = v18;
            v13 = v19;
            if ( !v19 )
            {
LABEL_14:
              j_j___libc_free_0((unsigned __int64)v8);
              return (__m128i *)a2;
            }
            goto LABEL_16;
          }
        }
        if ( !*(_QWORD *)(v14 + 24) )
        {
          v13 = v14;
          goto LABEL_7;
        }
      }
LABEL_16:
      v15 = a2 != 0;
      goto LABEL_17;
    }
    if ( v11 == v12 )
    {
      v23 = v8[2].m128i_u32[2];
      v24 = *(_DWORD *)(a2 + 40);
      if ( v23 < v24 )
        goto LABEL_3;
    }
    else
    {
      if ( v11 > v12 )
      {
LABEL_10:
        v25 = *(_QWORD *)(a2 + 32);
        if ( a1[4] != a2 )
        {
          v16 = sub_220EEE0(a2);
          v7 = a1 + 1;
          v17 = v25;
          if ( v11 < *(_QWORD *)(v16 + 32)
            || (v10 = (unsigned __int64 *)&v8[2], v11 == *(_QWORD *)(v16 + 32))
            && v8[2].m128i_i32[2] < *(_DWORD *)(v16 + 40) )
          {
            if ( *(_QWORD *)(a2 + 24) )
            {
              v13 = v16;
              v21 = 1;
              goto LABEL_20;
            }
            goto LABEL_33;
          }
          goto LABEL_13;
        }
        a2 = 0;
        goto LABEL_16;
      }
      v23 = v8[2].m128i_u32[2];
      v24 = *(_DWORD *)(a2 + 40);
    }
    if ( v23 <= v24 )
      goto LABEL_14;
    goto LABEL_10;
  }
  if ( !a1[5] )
    goto LABEL_13;
  v13 = a1[4];
  v22 = v8[2].m128i_u64[0];
  if ( *(_QWORD *)(v13 + 32) >= v22 )
  {
    if ( *(_QWORD *)(v13 + 32) == v22 && *(_DWORD *)(v13 + 40) < v8[2].m128i_i32[2] )
    {
      v15 = 0;
      goto LABEL_17;
    }
    goto LABEL_13;
  }
LABEL_7:
  v15 = 0;
LABEL_17:
  if ( v7 != (_QWORD *)v13 && !v15 )
  {
    v11 = v8[2].m128i_u64[0];
    v17 = *(_QWORD *)(v13 + 32);
    v21 = 1;
    if ( v17 > v11 )
      goto LABEL_20;
LABEL_33:
    v21 = 0;
    if ( v11 == v17 )
      v21 = v8[2].m128i_i32[2] < *(_DWORD *)(v13 + 40);
    goto LABEL_20;
  }
  v21 = 1;
LABEL_20:
  sub_220F040(v21, (__int64)v8, (_QWORD *)v13, v7);
  ++a1[5];
  return v8;
}
