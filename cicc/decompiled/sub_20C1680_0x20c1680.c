// Function: sub_20C1680
// Address: 0x20c1680
//
void __fastcall sub_20C1680(
        __m128i a1,
        double a2,
        __m128i a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        _QWORD *a9)
{
  bool v9; // cc
  char v12; // al
  __int16 v13; // dx
  __m128i *v14; // rsi
  __int64 v15; // rbx
  __int64 *v16; // rcx
  __int64 v17; // r15
  int v18; // edi
  int v19; // r8d
  __int64 v20; // r9
  __int64 v21; // rdx
  _QWORD *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rsi
  unsigned __int8 v25; // r10
  __int64 v26; // r8
  __int64 *v27; // rax
  __m128i *v28; // rsi
  __int32 v29; // edx
  __int64 v30; // rsi
  __int64 v31; // rax
  unsigned int v32; // edx
  __int64 *v33; // rsi
  __int64 v34; // rsi
  __int64 v35; // rax
  __int32 v36; // edx
  __int64 v37; // rax
  __int64 v38; // [rsp+8h] [rbp-78h]
  unsigned __int8 v39; // [rsp+17h] [rbp-69h]
  __int64 v40; // [rsp+18h] [rbp-68h]
  __m128i v41; // [rsp+20h] [rbp-60h] BYREF
  __int64 v42; // [rsp+30h] [rbp-50h] BYREF
  int v43; // [rsp+38h] [rbp-48h]
  __m128i v44; // [rsp+40h] [rbp-40h] BYREF

  v9 = *(_QWORD *)(a7 + 8) <= 1u;
  v41.m128i_i64[0] = a5;
  v41.m128i_i64[1] = a6;
  if ( !v9 )
    return;
  v12 = **(_BYTE **)a7;
  if ( v12 == 105 )
    goto LABEL_13;
  if ( v12 > 105 )
  {
    if ( v12 != 110 && v12 != 115 )
      return;
LABEL_13:
    a5 = v41.m128i_i64[0];
    v13 = *(_WORD *)(v41.m128i_i64[0] + 24);
    goto LABEL_14;
  }
  if ( v12 != 88 )
    return;
  v13 = *(_WORD *)(a5 + 24);
  if ( v13 == 5 )
  {
    v14 = *(__m128i **)(a8 + 8);
    if ( v14 == *(__m128i **)(a8 + 16) )
    {
      sub_1D4B0A0((const __m128i **)a8, v14, &v41);
    }
    else
    {
      if ( v14 )
      {
        *v14 = _mm_load_si128(&v41);
        v14 = *(__m128i **)(a8 + 8);
      }
      *(_QWORD *)(a8 + 8) = v14 + 1;
    }
    return;
  }
LABEL_14:
  if ( v13 == 32 || v13 == 10 )
  {
    if ( v13 == 12 )
    {
      if ( v12 == 110 )
        return;
      v20 = *(_QWORD *)(a5 + 96);
      v17 = a5;
      v15 = a5;
      goto LABEL_31;
    }
    v15 = a5;
  }
  else
  {
    v15 = 0;
    if ( v13 == 12 )
    {
      if ( v12 == 110 )
        return;
      v20 = *(_QWORD *)(a5 + 96);
      goto LABEL_59;
    }
  }
  if ( (unsigned __int16)(v13 - 34) <= 1u || v13 == 13 )
  {
    if ( v13 == 52 )
    {
LABEL_20:
      v16 = *(__int64 **)(a5 + 32);
      v15 = v16[5];
      v17 = *v16;
      v18 = *(unsigned __int16 *)(v15 + 24);
      v19 = *(unsigned __int16 *)(*v16 + 24);
      if ( (_WORD)v18 != 10 && v18 != 32 || (unsigned __int16)(v19 - 12) > 1u && (unsigned __int16)(v19 - 34) > 1u )
      {
        if ( v19 != 32 && v19 != 10
          || v18 != 12 && ((unsigned __int16)(*(_WORD *)(v15 + 24) - 34) > 1u && v18 != 13 || !v17) )
        {
          return;
        }
        v17 = v16[5];
        v15 = *v16;
      }
      if ( v12 == 110 )
        return;
      v20 = *(_QWORD *)(v17 + 96);
      goto LABEL_31;
    }
    if ( v12 == 110 )
      return;
    v20 = *(_QWORD *)(a5 + 96);
    if ( v15 )
    {
      v17 = a5;
LABEL_31:
      v21 = *(_QWORD *)(v15 + 88);
      v22 = *(_QWORD **)(v21 + 24);
      if ( *(_DWORD *)(v21 + 32) > 0x40u )
        v22 = (_QWORD *)*v22;
      v20 += (__int64)v22;
      v23 = *(_QWORD *)(a5 + 40) + 16LL * v41.m128i_u32[2];
      v24 = *(_QWORD *)(v15 + 72);
      v25 = *(_BYTE *)v23;
      v26 = *(_QWORD *)(v23 + 8);
      v42 = v24;
      if ( v24 )
      {
        v38 = v26;
        v39 = v25;
        v40 = v20;
        sub_1623A60((__int64)&v42, v24, 2);
        v26 = v38;
        v25 = v39;
        v20 = v40;
      }
      v43 = *(_DWORD *)(v15 + 64);
LABEL_36:
      v27 = sub_1D29600(a9, *(_QWORD *)(v17 + 88), (__int64)&v42, v25, v26, v20, 1, 0);
      v28 = *(__m128i **)(a8 + 8);
      v44.m128i_i64[0] = (__int64)v27;
      v44.m128i_i32[2] = v29;
      if ( v28 != *(__m128i **)(a8 + 16) )
      {
        if ( !v28 )
        {
LABEL_39:
          *(_QWORD *)(a8 + 8) = v28 + 1;
          goto LABEL_40;
        }
LABEL_38:
        *v28 = _mm_loadu_si128(&v44);
        v28 = *(__m128i **)(a8 + 8);
        goto LABEL_39;
      }
      goto LABEL_63;
    }
LABEL_59:
    v17 = a5;
    v37 = *(_QWORD *)(a5 + 40) + 16LL * v41.m128i_u32[2];
    v25 = *(_BYTE *)v37;
    v26 = *(_QWORD *)(v37 + 8);
    v42 = 0;
    v43 = 0;
    goto LABEL_36;
  }
  if ( v13 == 52 )
    goto LABEL_20;
  if ( !v15 || v12 == 115 )
    return;
  v30 = *(_QWORD *)(v15 + 72);
  v42 = v30;
  if ( v30 )
    sub_1623A60((__int64)&v42, v30, 2);
  v43 = *(_DWORD *)(v15 + 64);
  v31 = *(_QWORD *)(v15 + 88);
  v32 = *(_DWORD *)(v31 + 32);
  v33 = *(__int64 **)(v31 + 24);
  if ( v32 <= 0x40 )
    v34 = (__int64)((_QWORD)v33 << (64 - (unsigned __int8)v32)) >> (64 - (unsigned __int8)v32);
  else
    v34 = *v33;
  v35 = sub_1D38BB0((__int64)a9, v34, (__int64)&v42, 6, 0, 1, a1, a2, a3, 0);
  v28 = *(__m128i **)(a8 + 8);
  v44.m128i_i64[0] = v35;
  v44.m128i_i32[2] = v36;
  if ( v28 != *(__m128i **)(a8 + 16) )
  {
    if ( !v28 )
      goto LABEL_39;
    goto LABEL_38;
  }
LABEL_63:
  sub_1D4B3A0((const __m128i **)a8, v28, &v44);
LABEL_40:
  if ( v42 )
    sub_161E7C0((__int64)&v42, v42);
}
