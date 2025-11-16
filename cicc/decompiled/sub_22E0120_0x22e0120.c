// Function: sub_22E0120
// Address: 0x22e0120
//
__int64 __fastcall sub_22E0120(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r13
  __m128i *v9; // rdx
  __m128i si128; // xmm0
  const char *v11; // rax
  size_t v12; // rdx
  _BYTE *v13; // rdi
  unsigned __int8 *v14; // rsi
  _BYTE *v15; // rax
  __int64 v16; // rax
  _BYTE *v18; // rax
  size_t v19; // [rsp+8h] [rbp-38h]

  v8 = *a2;
  v9 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v9 <= 0x19u )
  {
    v8 = sub_CB6200(*a2, "Region Tree for function: ", 0x1Au);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_428CBE0);
    qmemcpy(&v9[1], "function: ", 10);
    *v9 = si128;
    *(_QWORD *)(v8 + 32) += 26LL;
  }
  v11 = sub_BD5D20(a3);
  v13 = *(_BYTE **)(v8 + 32);
  v14 = (unsigned __int8 *)v11;
  v15 = *(_BYTE **)(v8 + 24);
  if ( v15 - v13 < v12 )
  {
    v8 = sub_CB6200(v8, v14, v12);
    v15 = *(_BYTE **)(v8 + 24);
    v13 = *(_BYTE **)(v8 + 32);
  }
  else if ( v12 )
  {
    v19 = v12;
    memcpy(v13, v14, v12);
    v18 = *(_BYTE **)(v8 + 24);
    v13 = (_BYTE *)(v19 + *(_QWORD *)(v8 + 32));
    *(_QWORD *)(v8 + 32) = v13;
    if ( v13 != v18 )
      goto LABEL_6;
LABEL_9:
    sub_CB6200(v8, (unsigned __int8 *)"\n", 1u);
    goto LABEL_7;
  }
  if ( v13 == v15 )
    goto LABEL_9;
LABEL_6:
  *v13 = 10;
  ++*(_QWORD *)(v8 + 32);
LABEL_7:
  v16 = sub_BC1CD0(a4, &unk_4FDBD00, a3);
  sub_22E0050(v16 + 8, *a2);
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
