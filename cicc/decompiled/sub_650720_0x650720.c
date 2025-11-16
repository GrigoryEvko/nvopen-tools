// Function: sub_650720
// Address: 0x650720
//
__int64 __fastcall sub_650720(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        int a7,
        _BOOL4 a8,
        __m128i *a9,
        int a10)
{
  __int64 v10; // r15
  char v13; // dl
  __int64 v14; // r14
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v22; // rax
  __int64 v23; // r10
  _BOOL8 v24; // rcx
  char v25; // al
  __int64 v26; // rax
  __m128i *v27; // rax
  char v28; // al
  char v29; // al
  __int64 v30; // rsi
  __int64 v31; // rdi
  __int64 v35; // [rsp+18h] [rbp-88h]
  __int64 v36; // [rsp+18h] [rbp-88h]
  __int64 v37; // [rsp+18h] [rbp-88h]
  __int64 v38; // [rsp+28h] [rbp-78h] BYREF
  __m128i v39; // [rsp+30h] [rbp-70h] BYREF
  __m128i v40; // [rsp+40h] [rbp-60h]
  __m128i v41; // [rsp+50h] [rbp-50h]
  __m128i v42; // [rsp+60h] [rbp-40h]

  v10 = a1;
  v13 = *(_BYTE *)(a1 + 80);
  v14 = *a2;
  if ( v13 == 16 )
  {
    v15 = *(__int64 **)(a1 + 88);
    v10 = *v15;
    v13 = *(_BYTE *)(*v15 + 80);
  }
  if ( v13 == 24 )
  {
    v10 = *(_QWORD *)(v10 + 88);
    v13 = *(_BYTE *)(v10 + 80);
  }
  v39 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
  v40 = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
  v41 = _mm_loadu_si128(&xmmword_4D04A20);
  v42 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
  if ( (v40.m128i_i8[1] & 0x40) == 0 )
  {
    v40.m128i_i8[0] &= ~0x80u;
    v40.m128i_i64[1] = 0;
  }
  v38 = qword_4D04A08;
  if ( v13 != 13 )
  {
    v22 = sub_6506C0(v10, &v38, dword_4F04C5C);
    v23 = v22;
    if ( a5 )
    {
      *(_BYTE *)(v22 + 40) |= 2u;
      *(_QWORD *)(v22 + 48) = a5;
    }
    else
    {
      *(_QWORD *)(v22 + 48) = a4;
    }
    if ( a9 )
    {
      if ( a10 )
      {
        v37 = v22;
        v27 = (__m128i *)sub_5CF190(a9);
        v23 = v37;
        a9 = v27;
      }
      v35 = v23;
      sub_5CEC90(a9, v23, 29);
      v23 = v35;
    }
    v36 = v23;
    sub_876960(v10, &v38, v23, *a6);
    *a6 = v36;
    if ( v14 )
      goto LABEL_9;
LABEL_25:
    v24 = a8;
    if ( !a8 && a3 )
    {
      v25 = *(_BYTE *)(a3 + 80);
      if ( v25 == 3 )
      {
        v28 = *(_BYTE *)(v10 + 80);
        if ( v28 != 3 && (dword_4F077C4 != 2 || (unsigned __int8)(v28 - 4) > 2u) )
        {
          a8 = 0;
          goto LABEL_29;
        }
      }
      else
      {
        a8 = 0;
        if ( dword_4F077C4 != 2 )
          goto LABEL_29;
        if ( (unsigned __int8)(v25 - 4) > 2u )
          goto LABEL_29;
        v29 = *(_BYTE *)(v10 + 80);
        if ( v29 != 3 && (unsigned __int8)(v29 - 4) > 2u )
          goto LABEL_29;
      }
      v30 = *(_QWORD *)(v10 + 88);
      v31 = *(_QWORD *)(a3 + 88);
      a8 = 1;
      if ( v31 != v30 )
        a8 = sub_8D97D0(v31, v30, 0, v24, a5) != 0;
    }
LABEL_29:
    v26 = sub_885A40(v10, 1, &v39, (unsigned int)dword_4F04C5C, a8);
    v19 = v26;
    if ( a7 )
    {
      *a2 = v26;
      sub_877E90(v26, 0);
      return v19;
    }
    goto LABEL_15;
  }
  if ( !v14 )
    goto LABEL_25;
LABEL_9:
  v16 = dword_4F077BC;
  if ( dword_4F077BC )
    v16 = (_DWORD)qword_4F077B4 == 0;
  if ( (unsigned int)sub_7D06D0(v14, a1, v16, 0) || (unsigned int)sub_5EE800(v10, v14, (__int64)&v38, v17, v18) )
    return 0;
  v19 = sub_87ECE0(v10, &v39.m128i_u64[1], (unsigned int)dword_4F04C5C);
  v20 = sub_887160(v19, v14, 0, 0);
  if ( *a2 != v20 )
  {
    *a2 = v20;
    sub_877E90(v20, 0);
  }
LABEL_15:
  sub_877E90(v19, 0);
  return v19;
}
