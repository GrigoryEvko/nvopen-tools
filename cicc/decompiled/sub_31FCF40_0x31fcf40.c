// Function: sub_31FCF40
// Address: 0x31fcf40
//
void __fastcall sub_31FCF40(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int16 v4; // r15
  __int64 v7; // rbx
  __int64 v8; // r8
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // rbx
  __m128i *v19; // rsi
  __int64 v20; // [rsp+8h] [rbp-68h]
  __m128i v21; // [rsp+10h] [rbp-60h] BYREF
  __m128i v22; // [rsp+20h] [rbp-50h] BYREF
  __m128i v23; // [rsp+30h] [rbp-40h] BYREF

  v7 = *a4;
  v8 = sub_3211F40(*a1, a3);
  v9 = *(_DWORD *)a2;
  if ( *(_DWORD *)a2 > 5u )
  {
    if ( v9 == 6 )
      goto LABEL_16;
    v10 = 0;
  }
  else
  {
    if ( v9 <= 2 )
    {
      if ( !v9 )
      {
        v20 = 0;
        v4 = 6;
        v10 = 0;
        goto LABEL_5;
      }
LABEL_16:
      BUG();
    }
    (*(void (__fastcall **)(__m128i *, _QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(*a1 + 8LL) + 504LL))(
      &v21,
      *(_QWORD *)(*a1 + 8LL),
      (unsigned int)v7,
      a3,
      v8);
    v10 = v22.m128i_i64[0];
    v8 = v21.m128i_i64[1];
    v4 = v21.m128i_i16[0];
    v20 = v22.m128i_i64[1];
  }
LABEL_5:
  v11 = *a1;
  v12 = a1[1];
  v13 = *(_QWORD *)(*a1 + 792LL);
  v21.m128i_i64[1] = v20;
  v21.m128i_i16[0] = v4;
  v22.m128i_i64[0] = v10;
  v22.m128i_i64[1] = v8;
  v14 = *(_QWORD *)(v11 + 16);
  v15 = *(_QWORD *)(v14 + 2480);
  v16 = v14 + 8;
  if ( !v15 )
    v15 = v16;
  v17 = sub_2E792C0(v12, v7, v15, 0);
  v18 = (_QWORD *)(*(_QWORD *)(a2 + 8) + 32 * v7);
  v23.m128i_i64[0] = v17;
  v23.m128i_i64[1] = (__int64)(v18[1] - *v18) >> 3;
  v19 = *(__m128i **)(v13 + 424);
  if ( v19 == *(__m128i **)(v13 + 432) )
  {
    sub_31FCD90(v13 + 416, v19, &v21);
  }
  else
  {
    if ( v19 )
    {
      *v19 = _mm_loadu_si128(&v21);
      v19[1] = _mm_loadu_si128(&v22);
      v19[2] = _mm_loadu_si128(&v23);
      v19 = *(__m128i **)(v13 + 424);
    }
    *(_QWORD *)(v13 + 424) = v19 + 3;
  }
}
