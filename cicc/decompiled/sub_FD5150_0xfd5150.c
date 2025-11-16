// Function: sub_FD5150
// Address: 0xfd5150
//
__int64 __fastcall sub_FD5150(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r15
  __m128i *v8; // rdx
  __m128i si128; // xmm0
  const char *v10; // rax
  size_t v11; // rdx
  _BYTE *v12; // rdi
  unsigned __int8 *v13; // rsi
  _BYTE *v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rbx
  __int64 v19; // r14
  __int64 v20; // rbx
  __int64 v21; // r15
  __int64 v22; // rdx
  _BYTE *v23; // rax
  _BYTE *v25; // rax
  size_t v26; // [rsp+0h] [rbp-40h]

  v7 = *a2;
  v8 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v8 <= 0x22u )
  {
    v7 = sub_CB6200(v7, "RegisterPressureInfo for function: ", 0x23u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8C0F0);
    v8[2].m128i_i8[2] = 32;
    v8[2].m128i_i16[0] = 14958;
    *v8 = si128;
    v8[1] = _mm_load_si128((const __m128i *)&xmmword_3F8C100);
    *(_QWORD *)(v7 + 32) += 35LL;
  }
  v10 = sub_BD5D20(a3);
  v12 = *(_BYTE **)(v7 + 32);
  v13 = (unsigned __int8 *)v10;
  v14 = *(_BYTE **)(v7 + 24);
  if ( v14 - v12 < v11 )
  {
    v7 = sub_CB6200(v7, v13, v11);
    v14 = *(_BYTE **)(v7 + 24);
    v12 = *(_BYTE **)(v7 + 32);
  }
  else if ( v11 )
  {
    v26 = v11;
    memcpy(v12, v13, v11);
    v25 = *(_BYTE **)(v7 + 24);
    v12 = (_BYTE *)(v26 + *(_QWORD *)(v7 + 32));
    *(_QWORD *)(v7 + 32) = v12;
    if ( v12 != v25 )
      goto LABEL_6;
    goto LABEL_16;
  }
  if ( v12 != v14 )
  {
LABEL_6:
    *v12 = 10;
    ++*(_QWORD *)(v7 + 32);
    goto LABEL_7;
  }
LABEL_16:
  sub_CB6200(v7, (unsigned __int8 *)"\n", 1u);
LABEL_7:
  v15 = sub_BC1CD0(a4, &unk_4F8D468, a3);
  v16 = sub_BC1CD0(a4, &unk_4F89C30, a3);
  v17 = v15 + 8;
  sub_FD3980(v15 + 8, v16 + 8);
  v18 = *(_QWORD *)(v15 + 8);
  v19 = *(_QWORD *)(v18 + 80);
  v20 = v18 + 72;
  v21 = *a2;
  if ( v19 != v20 )
  {
    do
    {
      v22 = v19 - 24;
      if ( !v19 )
        v22 = 0;
      sub_FCE270(v17, v21, v22, 0);
      v19 = *(_QWORD *)(v19 + 8);
    }
    while ( v20 != v19 );
    v21 = *a2;
  }
  v23 = *(_BYTE **)(v21 + 32);
  if ( *(_BYTE **)(v21 + 24) == v23 )
  {
    sub_CB6200(v21, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v23 = 10;
    ++*(_QWORD *)(v21 + 32);
  }
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
