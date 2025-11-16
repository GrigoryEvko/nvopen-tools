// Function: sub_22A4610
// Address: 0x22a4610
//
__int64 __fastcall sub_22A4610(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r13
  __m128i *v9; // rdx
  const char *v10; // rax
  size_t v11; // rdx
  _BYTE *v12; // rdi
  unsigned __int8 *v13; // rsi
  _BYTE *v14; // rax
  __int64 v15; // rax
  _BYTE *v17; // rax
  size_t v18; // [rsp+8h] [rbp-38h]

  v8 = *a2;
  v9 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v9 <= 0x1Fu )
  {
    v8 = sub_CB6200(*a2, "DominanceFrontier for function: ", 0x20u);
  }
  else
  {
    *v9 = _mm_load_si128((const __m128i *)&xmmword_4289C60);
    v9[1] = _mm_load_si128((const __m128i *)&xmmword_4289C70);
    *(_QWORD *)(v8 + 32) += 32LL;
  }
  v10 = sub_BD5D20(a3);
  v12 = *(_BYTE **)(v8 + 32);
  v13 = (unsigned __int8 *)v10;
  v14 = *(_BYTE **)(v8 + 24);
  if ( v14 - v12 < v11 )
  {
    v8 = sub_CB6200(v8, v13, v11);
    v14 = *(_BYTE **)(v8 + 24);
    v12 = *(_BYTE **)(v8 + 32);
  }
  else if ( v11 )
  {
    v18 = v11;
    memcpy(v12, v13, v11);
    v17 = *(_BYTE **)(v8 + 24);
    v12 = (_BYTE *)(v18 + *(_QWORD *)(v8 + 32));
    *(_QWORD *)(v8 + 32) = v12;
    if ( v12 != v17 )
      goto LABEL_6;
LABEL_9:
    sub_CB6200(v8, (unsigned __int8 *)"\n", 1u);
    goto LABEL_7;
  }
  if ( v12 == v14 )
    goto LABEL_9;
LABEL_6:
  *v12 = 10;
  ++*(_QWORD *)(v8 + 32);
LABEL_7:
  v15 = sub_BC1CD0(a4, &unk_4FDB678, a3);
  sub_22A3EC0(v15 + 8, *a2);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
