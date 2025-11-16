// Function: sub_30B1BF0
// Address: 0x30b1bf0
//
__int64 __fastcall sub_30B1BF0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v9; // r13
  __m128i *v10; // rdx
  const char *v11; // rax
  size_t v12; // rdx
  _BYTE *v13; // rdi
  unsigned __int8 *v14; // rsi
  unsigned __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // rax
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  size_t v21; // [rsp+0h] [rbp-40h]

  v9 = *a2;
  v10 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v10 <= 0xFu )
  {
    v9 = sub_CB6200(v9, "'DDG' for loop '", 0x10u);
  }
  else
  {
    *v10 = _mm_load_si128((const __m128i *)&xmmword_44CBB10);
    *(_QWORD *)(v9 + 32) += 16LL;
  }
  v11 = sub_BD5D20(**(_QWORD **)(a3 + 32));
  v13 = *(_BYTE **)(v9 + 32);
  v14 = (unsigned __int8 *)v11;
  v15 = *(_QWORD *)(v9 + 24) - (_QWORD)v13;
  if ( v15 < v12 )
  {
    v20 = sub_CB6200(v9, v14, v12);
    v13 = *(_BYTE **)(v20 + 32);
    v9 = v20;
    v15 = *(_QWORD *)(v20 + 24) - (_QWORD)v13;
  }
  else if ( v12 )
  {
    v21 = v12;
    memcpy(v13, v14, v12);
    v13 = (_BYTE *)(v21 + *(_QWORD *)(v9 + 32));
    v19 = *(_QWORD *)(v9 + 24) - (_QWORD)v13;
    *(_QWORD *)(v9 + 32) = v13;
    if ( v19 > 2 )
      goto LABEL_6;
LABEL_9:
    sub_CB6200(v9, "':\n", 3u);
    goto LABEL_7;
  }
  if ( v15 <= 2 )
    goto LABEL_9;
LABEL_6:
  v13[2] = 10;
  *(_WORD *)v13 = 14887;
  *(_QWORD *)(v9 + 32) += 3LL;
LABEL_7:
  v16 = *a2;
  v17 = sub_22D3D20(a4, qword_502E9E0, (__int64 *)a3, a5);
  sub_30B1A90(v16, *(_QWORD *)(v17 + 8));
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
