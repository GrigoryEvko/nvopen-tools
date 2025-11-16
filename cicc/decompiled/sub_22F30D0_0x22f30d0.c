// Function: sub_22F30D0
// Address: 0x22f30d0
//
__int64 __fastcall sub_22F30D0(__int64 a1, __int64 *a2, __int64 a3, __m128i a4)
{
  char v7; // bl
  char *v8; // rax
  __int64 v9; // rdx
  char v11; // al
  size_t v12; // rdx
  unsigned __int8 *v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // r8
  _BYTE *v16; // rax
  __int64 v17; // rax
  void *v18; // rdx
  __int64 v19; // r14
  const char *v20; // rax
  size_t v21; // rdx
  _WORD *v22; // rdi
  unsigned __int8 *v23; // rsi
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  _WORD *v27; // rdx
  size_t v28; // [rsp+8h] [rbp-28h]

  v7 = *(_BYTE *)(a3 + 128);
  sub_B2B9F0(a3, unk_4F81788);
  v8 = (char *)sub_BD5D20(a3);
  if ( sub_BC63A0(v8, v9) )
  {
    v11 = sub_BC5DE0();
    v12 = a2[2];
    v13 = (unsigned __int8 *)a2[1];
    v14 = *a2;
    if ( v11 )
    {
      v17 = sub_CB6200(v14, v13, v12);
      v18 = *(void **)(v17 + 32);
      v19 = v17;
      if ( *(_QWORD *)(v17 + 24) - (_QWORD)v18 <= 0xBu )
      {
        v19 = sub_CB6200(v17, " (function: ", 0xCu);
      }
      else
      {
        qmemcpy(v18, " (function: ", 12);
        *(_QWORD *)(v17 + 32) += 12LL;
      }
      v20 = sub_BD5D20(a3);
      v22 = *(_WORD **)(v19 + 32);
      v23 = (unsigned __int8 *)v20;
      v24 = *(_QWORD *)(v19 + 24) - (_QWORD)v22;
      if ( v24 < v21 )
      {
        v25 = sub_CB6200(v19, v23, v21);
        v22 = *(_WORD **)(v25 + 32);
        v19 = v25;
        v24 = *(_QWORD *)(v25 + 24) - (_QWORD)v22;
      }
      else if ( v21 )
      {
        v28 = v21;
        memcpy(v22, v23, v21);
        v26 = *(_QWORD *)(v19 + 24);
        v27 = (_WORD *)(*(_QWORD *)(v19 + 32) + v28);
        *(_QWORD *)(v19 + 32) = v27;
        v22 = v27;
        v24 = v26 - (_QWORD)v27;
      }
      if ( v24 <= 1 )
      {
        v19 = sub_CB6200(v19, (unsigned __int8 *)")\n", 2u);
      }
      else
      {
        *v22 = 2601;
        *(_QWORD *)(v19 + 32) += 2LL;
      }
      sub_A69980(*(__int64 (__fastcall ***)())(a3 + 40), v19, 0, 0, 0, a4);
    }
    else
    {
      v15 = sub_CB6200(v14, v13, v12);
      v16 = *(_BYTE **)(v15 + 32);
      if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 24) )
      {
        v15 = sub_CB5D20(v15, 10);
      }
      else
      {
        *(_QWORD *)(v15 + 32) = v16 + 1;
        *v16 = 10;
      }
      sub_A69870(a3, (_BYTE *)v15, 0);
    }
  }
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
  sub_B2B9F0(a3, v7);
  return a1;
}
