// Function: sub_B3A6A0
// Address: 0xb3a6a0
//
__int64 __fastcall sub_B3A6A0(_QWORD *a1, __int64 a2, __m128i a3)
{
  char v5; // bl
  __int64 v6; // rdi
  __int64 v7; // rdx
  char v9; // al
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // r8
  _BYTE *v14; // rax
  __int64 v15; // rax
  void *v16; // rdx
  __int64 v17; // r13
  __int64 v18; // rax
  size_t v19; // rdx
  _WORD *v20; // rdi
  const void *v21; // rsi
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  _WORD *v25; // rdx
  size_t v26; // [rsp+8h] [rbp-28h]

  v5 = *(_BYTE *)(a2 + 128);
  sub_B2B9F0(a2, unk_4F81788);
  v6 = sub_BD5D20(a2);
  if ( (unsigned __int8)sub_BC63A0(v6, v7) )
  {
    v9 = sub_BC5DE0();
    v10 = a1[24];
    v11 = a1[23];
    v12 = a1[22];
    if ( v9 )
    {
      v15 = sub_CB6200(v12, v11, v10);
      v16 = *(void **)(v15 + 32);
      v17 = v15;
      if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 <= 0xBu )
      {
        v17 = sub_CB6200(v15, " (function: ", 12);
      }
      else
      {
        qmemcpy(v16, " (function: ", 12);
        *(_QWORD *)(v15 + 32) += 12LL;
      }
      v18 = sub_BD5D20(a2);
      v20 = *(_WORD **)(v17 + 32);
      v21 = (const void *)v18;
      v22 = *(_QWORD *)(v17 + 24) - (_QWORD)v20;
      if ( v22 < v19 )
      {
        v23 = sub_CB6200(v17, v21, v19);
        v20 = *(_WORD **)(v23 + 32);
        v17 = v23;
        v22 = *(_QWORD *)(v23 + 24) - (_QWORD)v20;
      }
      else if ( v19 )
      {
        v26 = v19;
        memcpy(v20, v21, v19);
        v24 = *(_QWORD *)(v17 + 24);
        v25 = (_WORD *)(*(_QWORD *)(v17 + 32) + v26);
        *(_QWORD *)(v17 + 32) = v25;
        v20 = v25;
        v22 = v24 - (_QWORD)v25;
      }
      if ( v22 <= 1 )
      {
        v17 = sub_CB6200(v17, ")\n", 2);
      }
      else
      {
        *v20 = 2601;
        *(_QWORD *)(v17 + 32) += 2LL;
      }
      sub_A69980(*(__int64 (__fastcall ***)())(a2 + 40), v17, 0, 0, 0, a3);
    }
    else
    {
      v13 = sub_CB6200(v12, v11, v10);
      v14 = *(_BYTE **)(v13 + 32);
      if ( (unsigned __int64)v14 >= *(_QWORD *)(v13 + 24) )
      {
        v13 = sub_CB5D20(v13, 10);
      }
      else
      {
        *(_QWORD *)(v13 + 32) = v14 + 1;
        *v14 = 10;
      }
      sub_A69870(a2, (_BYTE *)v13, 0);
    }
  }
  sub_B2B9F0(a2, v5);
  return 0;
}
