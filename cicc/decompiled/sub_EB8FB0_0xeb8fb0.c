// Function: sub_EB8FB0
// Address: 0xeb8fb0
//
__int64 __fastcall sub_EB8FB0(__int64 a1)
{
  __int64 v2; // r12
  _DWORD *v3; // rax
  __int64 v4; // r14
  const __m128i *v5; // rsi
  _DWORD *v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __m128i v13; // kr00_16
  int v14; // eax
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned int v18; // eax
  __int64 *v19; // rdi
  unsigned int v20; // r12d
  __int64 v21; // rax
  const char *v22; // rax
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v28; // [rsp+18h] [rbp-A8h] BYREF
  __m128i v29; // [rsp+20h] [rbp-A0h] BYREF
  __m128i v30; // [rsp+30h] [rbp-90h] BYREF
  const __m128i *v31; // [rsp+40h] [rbp-80h] BYREF
  __m128i *v32; // [rsp+48h] [rbp-78h]
  const __m128i *v33; // [rsp+50h] [rbp-70h]
  __m128i v34[2]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v35; // [rsp+80h] [rbp-40h]

  v2 = 0;
  v3 = *(_DWORD **)(a1 + 48);
  v31 = 0;
  v32 = 0;
  v33 = 0;
  if ( *v3 == 2 )
  {
    v4 = a1 + 40;
    while ( 1 )
    {
      v7 = sub_ECD690(v4);
      v29 = 0u;
      v2 = v7;
      if ( (unsigned __int8)sub_EB61F0(a1, v29.m128i_i64) )
        break;
      v8 = *(_QWORD *)(a1 + 224);
      v35 = 261;
      v34[0] = v29;
      v9 = sub_E6C460(v8, (const char **)v34);
      v10 = sub_ECD690(v4);
      v30 = 0u;
      v2 = v10;
      if ( (unsigned __int8)sub_EB61F0(a1, v30.m128i_i64) )
        break;
      v11 = *(_QWORD *)(a1 + 224);
      v35 = 261;
      v34[0] = v30;
      v12 = sub_E6C460(v11, (const char **)v34);
      v34[0].m128i_i64[0] = v9;
      v5 = v32;
      v34[0].m128i_i64[1] = v12;
      if ( v32 == v33 )
      {
        sub_EA9B60(&v31, v32, v34);
        if ( **(_DWORD **)(a1 + 48) != 2 )
          goto LABEL_10;
      }
      else
      {
        if ( v32 )
        {
          *v32 = _mm_loadu_si128(v34);
          v5 = v32;
        }
        v6 = *(_DWORD **)(a1 + 48);
        v32 = (__m128i *)&v5[1];
        if ( *v6 != 2 )
          goto LABEL_10;
      }
    }
    HIBYTE(v35) = 1;
    v22 = "expected identifier in directive";
    goto LABEL_22;
  }
LABEL_10:
  v35 = 259;
  v30 = 0u;
  v34[0].m128i_i64[0] = (__int64)"expected comma before def_range type in .cv_def_range directive";
  if ( (unsigned __int8)sub_ECE210(a1, 26, v34) || (unsigned __int8)sub_EB61F0(a1, v30.m128i_i64) )
  {
    HIBYTE(v35) = 1;
    v22 = "expected def_range type in directive";
LABEL_22:
    v34[0].m128i_i64[0] = (__int64)v22;
    LOBYTE(v35) = 3;
    v20 = sub_ECDA70(a1, v2, v34, 0, 0);
    goto LABEL_23;
  }
  v13 = v30;
  v14 = sub_C92610();
  v15 = sub_C92860((__int64 *)(a1 + 896), (const void *)v13.m128i_i64[0], v13.m128i_u64[1], v14);
  if ( v15 == -1 )
    goto LABEL_27;
  v16 = *(_QWORD *)(a1 + 896);
  v17 = v16 + 8LL * v15;
  if ( v17 == v16 + 8LL * *(unsigned int *)(a1 + 904) )
    goto LABEL_27;
  v18 = *(_DWORD *)(*(_QWORD *)v17 + 8LL);
  if ( v18 != 3 )
  {
    if ( v18 <= 3 )
    {
      if ( v18 != 1 )
      {
        if ( v18 == 2 )
        {
          v35 = 259;
          v34[0].m128i_i64[0] = (__int64)"expected comma before offset in .cv_def_range directive";
          if ( !(unsigned __int8)sub_ECE210(a1, 26, v34) && !(unsigned __int8)sub_EAC8B0(a1, &v29) )
          {
            v19 = *(__int64 **)(a1 + 232);
            v20 = 1;
            v21 = *v19;
            v34[0].m128i_i32[0] = v29.m128i_i32[0];
            (*(void (__fastcall **)(__int64 *, const __m128i *, signed __int64))(v21 + 792))(v19, v31, v32 - v31);
            goto LABEL_23;
          }
          goto LABEL_41;
        }
        goto LABEL_27;
      }
      v35 = 259;
      v34[0].m128i_i64[0] = (__int64)"expected comma before register number in .cv_def_range directive";
      if ( !(unsigned __int8)sub_ECE210(a1, 26, v34) && !(unsigned __int8)sub_EAC8B0(a1, &v29) )
      {
        v20 = 1;
        v24 = *(_QWORD *)(a1 + 232);
        v34[0].m128i_i32[0] = v29.m128i_u16[0];
        (*(void (__fastcall **)(__int64, const __m128i *, signed __int64, _QWORD))(*(_QWORD *)v24 + 784LL))(
          v24,
          v31,
          v32 - v31,
          v29.m128i_u16[0]);
        goto LABEL_23;
      }
LABEL_35:
      HIBYTE(v35) = 1;
      v22 = "expected register number";
      goto LABEL_22;
    }
    if ( v18 == 4 )
    {
      v35 = 259;
      v34[0].m128i_i64[0] = (__int64)"expected comma before register number in .cv_def_range directive";
      if ( (unsigned __int8)sub_ECE210(a1, 26, v34) || (unsigned __int8)sub_EAC8B0(a1, &v27) )
      {
        HIBYTE(v35) = 1;
        v22 = "expected register value";
        goto LABEL_22;
      }
      v35 = 259;
      v34[0].m128i_i64[0] = (__int64)"expected comma before flag value in .cv_def_range directive";
      if ( (unsigned __int8)sub_ECE210(a1, 26, v34) || (unsigned __int8)sub_EAC8B0(a1, &v28) )
      {
        HIBYTE(v35) = 1;
        v22 = "expected flag value";
        goto LABEL_22;
      }
      v35 = 259;
      v34[0].m128i_i64[0] = (__int64)"expected comma before base pointer offset in .cv_def_range directive";
      if ( (unsigned __int8)sub_ECE210(a1, 26, v34) || (unsigned __int8)sub_EAC8B0(a1, &v29) )
      {
        HIBYTE(v35) = 1;
        v22 = "expected base pointer offset value";
        goto LABEL_22;
      }
      v20 = 1;
      v26 = *(_QWORD *)(a1 + 232);
      v34[0].m128i_i16[0] = v27;
      v34[0].m128i_i16[1] = v28;
      v34[0].m128i_i32[1] = v29.m128i_i32[0];
      (*(void (__fastcall **)(__int64, const __m128i *, signed __int64, __int64))(*(_QWORD *)v26 + 768LL))(
        v26,
        v31,
        v32 - v31,
        v34[0].m128i_i64[0]);
      goto LABEL_23;
    }
LABEL_27:
    HIBYTE(v35) = 1;
    v22 = "unexpected def_range type in .cv_def_range directive";
    goto LABEL_22;
  }
  v35 = 259;
  v34[0].m128i_i64[0] = (__int64)"expected comma before register number in .cv_def_range directive";
  if ( (unsigned __int8)sub_ECE210(a1, 26, v34) || (unsigned __int8)sub_EAC8B0(a1, &v28) )
    goto LABEL_35;
  v35 = 259;
  v34[0].m128i_i64[0] = (__int64)"expected comma before offset in .cv_def_range directive";
  if ( (unsigned __int8)sub_ECE210(a1, 26, v34) || (unsigned __int8)sub_EAC8B0(a1, &v29) )
  {
LABEL_41:
    HIBYTE(v35) = 1;
    v22 = "expected offset value";
    goto LABEL_22;
  }
  v20 = 1;
  v25 = *(_QWORD *)(a1 + 232);
  v34[0].m128i_i32[0] = (unsigned __int16)v28;
  v34[0].m128i_i32[1] = v29.m128i_i32[0];
  (*(void (__fastcall **)(__int64, const __m128i *, signed __int64, __int64))(*(_QWORD *)v25 + 776LL))(
    v25,
    v31,
    v32 - v31,
    v34[0].m128i_i64[0]);
LABEL_23:
  if ( v31 )
    j_j___libc_free_0(v31, (char *)v33 - (char *)v31);
  return v20;
}
