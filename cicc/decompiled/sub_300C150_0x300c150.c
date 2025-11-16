// Function: sub_300C150
// Address: 0x300c150
//
_BYTE *__fastcall sub_300C150(__int64 *a1, __int64 a2)
{
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  _BYTE *result; // rax
  int i; // ebx
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 *v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rdx
  _DWORD *v13; // rdx
  _WORD *v14; // rdx
  char *v15; // rsi
  size_t v16; // rax
  _BYTE *v17; // rdi
  size_t v18; // rdx
  _BYTE *v19; // rax
  int v20; // r13d
  __int64 v21; // r10
  __int64 v22; // r10
  __int64 v23; // rdx
  __int64 v24; // rax
  _WORD *v25; // rdx
  __int64 v26; // r8
  char *v27; // rsi
  size_t v28; // rax
  size_t v29; // rdx
  _BYTE *v30; // rax
  _BYTE *v31; // rdi
  __int64 v32; // rbx
  size_t v33; // [rsp+0h] [rbp-B0h]
  __int64 v34; // [rsp+8h] [rbp-A8h]
  __int64 v35; // [rsp+10h] [rbp-A0h]
  __int64 v36; // [rsp+10h] [rbp-A0h]
  size_t v37; // [rsp+10h] [rbp-A0h]
  int v38; // [rsp+1Ch] [rbp-94h]
  int v39; // [rsp+1Ch] [rbp-94h]
  __int64 v40[2]; // [rsp+20h] [rbp-90h] BYREF
  void (__fastcall *v41)(__int64 *, __int64 *, __int64); // [rsp+30h] [rbp-80h]
  void (__fastcall *v42)(__int64 *, __int64); // [rsp+38h] [rbp-78h]
  __int64 v43[2]; // [rsp+40h] [rbp-70h] BYREF
  void (__fastcall *v44)(__int64 *, __int64 *, __int64); // [rsp+50h] [rbp-60h]
  void (__fastcall *v45)(__int64 *, __int64); // [rsp+58h] [rbp-58h]
  __int64 v46[2]; // [rsp+60h] [rbp-50h] BYREF
  void (__fastcall *v47)(__int64 *, __int64 *, __int64); // [rsp+70h] [rbp-40h]
  void (__fastcall *v48)(__int64 *, __int64); // [rsp+78h] [rbp-38h]

  v4 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 0x22u )
  {
    sub_CB6200(a2, "********** REGISTER MAP **********\n", 0x23u);
    result = *(_BYTE **)(a2 + 32);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42F4EF0);
    v4[2].m128i_i8[2] = 10;
    v4[2].m128i_i16[0] = 10794;
    *v4 = si128;
    v4[1] = _mm_load_si128((const __m128i *)&xmmword_42F4F00);
    result = (_BYTE *)(*(_QWORD *)(a2 + 32) + 35LL);
    *(_QWORD *)(a2 + 32) = result;
  }
  v38 = *(_DWORD *)(*a1 + 64);
  if ( !v38 )
    goto LABEL_50;
  for ( i = 0; i != v38; ++i )
  {
    while ( 1 )
    {
      v8 = i & 0x7FFFFFFF;
      if ( *(_DWORD *)(a1[4] + 4 * v8) )
        break;
      if ( ++i == v38 )
        goto LABEL_26;
    }
    if ( *(_QWORD *)(a2 + 24) <= (unsigned __int64)result )
    {
      v9 = sub_CB5D20(a2, 91);
    }
    else
    {
      v9 = a2;
      *(_QWORD *)(a2 + 32) = result + 1;
      *result = 91;
    }
    v10 = v40;
    v11 = i | 0x80000000;
    sub_2FF6320(v40, v11, a1[2], 0, 0);
    if ( !v41
      || ((v42(v40, v9), v13 = *(_DWORD **)(v9 + 32), *(_QWORD *)(v9 + 24) - (_QWORD)v13 <= 3u)
        ? (v9 = sub_CB6200(v9, (unsigned __int8 *)" -> ", 4u))
        : (*v13 = 540945696, *(_QWORD *)(v9 + 32) += 4LL),
          v11 = *(unsigned int *)(a1[4] + 4 * v8),
          v10 = v43,
          sub_2FF6320(v43, v11, a1[2], 0, 0),
          !v44) )
    {
LABEL_62:
      sub_4263D6(v10, v11, v12);
    }
    v45(v43, v9);
    v14 = *(_WORD **)(v9 + 32);
    if ( *(_QWORD *)(v9 + 24) - (_QWORD)v14 <= 1u )
    {
      v9 = sub_CB6200(v9, (unsigned __int8 *)"] ", 2u);
    }
    else
    {
      *v14 = 8285;
      *(_QWORD *)(v9 + 32) += 2LL;
    }
    v15 = (char *)(*(_QWORD *)(a1[2] + 80)
                 + *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 56) + 16 * v8) & 0xFFFFFFFFFFFFFFF8LL)
                                   + 16LL));
    if ( !v15 )
      goto LABEL_48;
    v16 = strlen(v15);
    v17 = *(_BYTE **)(v9 + 32);
    v18 = v16;
    v19 = *(_BYTE **)(v9 + 24);
    if ( v18 > v19 - v17 )
    {
      v9 = sub_CB6200(v9, (unsigned __int8 *)v15, v18);
LABEL_48:
      v17 = *(_BYTE **)(v9 + 32);
      if ( v17 != *(_BYTE **)(v9 + 24) )
        goto LABEL_20;
      goto LABEL_49;
    }
    if ( v18 )
    {
      v33 = v18;
      memcpy(v17, v15, v18);
      v19 = *(_BYTE **)(v9 + 24);
      v17 = (_BYTE *)(v33 + *(_QWORD *)(v9 + 32));
      *(_QWORD *)(v9 + 32) = v17;
    }
    if ( v17 != v19 )
    {
LABEL_20:
      *v17 = 10;
      ++*(_QWORD *)(v9 + 32);
      goto LABEL_21;
    }
LABEL_49:
    sub_CB6200(v9, (unsigned __int8 *)"\n", 1u);
LABEL_21:
    if ( v44 )
      v44(v43, v43, 3);
    if ( v41 )
      v41(v40, v40, 3);
    result = *(_BYTE **)(a2 + 32);
  }
LABEL_26:
  v39 = *(_DWORD *)(*a1 + 64);
  if ( v39 )
  {
    v20 = 0;
    while ( 1 )
    {
      v32 = v20 & 0x7FFFFFFF;
      if ( *(_DWORD *)(a1[7] + 4 * v32) != 0x7FFFFFFF )
        break;
LABEL_43:
      if ( ++v20 == v39 )
        goto LABEL_50;
    }
    if ( *(_QWORD *)(a2 + 24) > (unsigned __int64)result )
    {
      v21 = a2;
      *(_QWORD *)(a2 + 32) = result + 1;
      *result = 91;
    }
    else
    {
      v21 = sub_CB5D20(a2, 91);
    }
    v11 = v20 | 0x80000000;
    v10 = v46;
    v35 = v21;
    sub_2FF6320(v46, v11, a1[2], 0, 0);
    if ( !v47 )
      goto LABEL_62;
    v48(v46, v35);
    v22 = v35;
    v23 = *(_QWORD *)(v35 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v35 + 24) - v23) <= 6 )
    {
      v22 = sub_CB6200(v35, " -> fi#", 7u);
    }
    else
    {
      *(_DWORD *)v23 = 540945696;
      *(_WORD *)(v23 + 4) = 26982;
      *(_BYTE *)(v23 + 6) = 35;
      *(_QWORD *)(v35 + 32) += 7LL;
    }
    v24 = sub_CB59F0(v22, *(int *)(a1[7] + 4 * v32));
    v25 = *(_WORD **)(v24 + 32);
    v26 = v24;
    if ( *(_QWORD *)(v24 + 24) - (_QWORD)v25 <= 1u )
    {
      v26 = sub_CB6200(v24, (unsigned __int8 *)"] ", 2u);
    }
    else
    {
      *v25 = 8285;
      *(_QWORD *)(v24 + 32) += 2LL;
    }
    v27 = (char *)(*(_QWORD *)(a1[2] + 80)
                 + *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 56) + 16 * v32) & 0xFFFFFFFFFFFFFFF8LL)
                                   + 16LL));
    if ( v27 )
    {
      v36 = v26;
      v28 = strlen(v27);
      v26 = v36;
      v29 = v28;
      v30 = *(_BYTE **)(v36 + 24);
      v31 = *(_BYTE **)(v36 + 32);
      if ( v29 <= v30 - v31 )
      {
        if ( v29 )
        {
          v34 = v36;
          v37 = v29;
          memcpy(v31, v27, v29);
          v26 = v34;
          v31 = (_BYTE *)(*(_QWORD *)(v34 + 32) + v37);
          v30 = *(_BYTE **)(v34 + 24);
          *(_QWORD *)(v34 + 32) = v31;
        }
        if ( v31 != v30 )
        {
LABEL_39:
          *v31 = 10;
          ++*(_QWORD *)(v26 + 32);
LABEL_40:
          if ( v47 )
            v47(v46, v46, 3);
          result = *(_BYTE **)(a2 + 32);
          goto LABEL_43;
        }
LABEL_54:
        sub_CB6200(v26, (unsigned __int8 *)"\n", 1u);
        goto LABEL_40;
      }
      v26 = sub_CB6200(v36, (unsigned __int8 *)v27, v29);
    }
    v31 = *(_BYTE **)(v26 + 32);
    if ( v31 != *(_BYTE **)(v26 + 24) )
      goto LABEL_39;
    goto LABEL_54;
  }
LABEL_50:
  if ( *(_QWORD *)(a2 + 24) <= (unsigned __int64)result )
    return (_BYTE *)sub_CB5D20(a2, 10);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 10;
  return result;
}
