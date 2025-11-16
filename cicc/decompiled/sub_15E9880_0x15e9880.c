// Function: sub_15E9880
// Address: 0x15e9880
//
__int64 __fastcall sub_15E9880(__int64 a1, __int64 *a2, __int64 a3, __m128i a4)
{
  __int64 v7; // rdi
  __int64 v8; // rdx
  char v10; // al
  __int64 v11; // rdx
  const char *v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rax
  void *v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rax
  size_t v18; // rdx
  _WORD *v19; // rdi
  const char *v20; // rsi
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  _WORD *v25; // rdx
  size_t v26; // [rsp+8h] [rbp-28h]

  v7 = sub_1649960(a3);
  if ( (unsigned __int8)sub_160E740(v7, v8) )
  {
    v10 = sub_160E720();
    v11 = a2[2];
    v12 = (const char *)a2[1];
    v13 = *a2;
    if ( v10 )
    {
      v14 = sub_16E7EE0(v13, v12, v11);
      v15 = *(void **)(v14 + 24);
      v16 = v14;
      if ( *(_QWORD *)(v14 + 16) - (_QWORD)v15 <= 0xBu )
      {
        v16 = sub_16E7EE0(v14, " (function: ", 12);
      }
      else
      {
        qmemcpy(v15, " (function: ", 12);
        *(_QWORD *)(v14 + 24) += 12LL;
      }
      v17 = sub_1649960(a3);
      v19 = *(_WORD **)(v16 + 24);
      v20 = (const char *)v17;
      v21 = *(_QWORD *)(v16 + 16) - (_QWORD)v19;
      if ( v21 < v18 )
      {
        v23 = sub_16E7EE0(v16, v20);
        v19 = *(_WORD **)(v23 + 24);
        v16 = v23;
        v21 = *(_QWORD *)(v23 + 16) - (_QWORD)v19;
      }
      else if ( v18 )
      {
        v26 = v18;
        memcpy(v19, v20, v18);
        v24 = *(_QWORD *)(v16 + 16);
        v25 = (_WORD *)(*(_QWORD *)(v16 + 24) + v26);
        *(_QWORD *)(v16 + 24) = v25;
        v19 = v25;
        v21 = v24 - (_QWORD)v25;
      }
      if ( v21 <= 1 )
      {
        v16 = sub_16E7EE0(v16, ")\n", 2);
      }
      else
      {
        *v19 = 2601;
        *(_QWORD *)(v16 + 24) += 2LL;
      }
      sub_155BB10(*(_QWORD *)(a3 + 40), v16, 0, 0, 0, a4);
    }
    else
    {
      v22 = sub_16E7EE0(v13, v12, v11);
      sub_155C2B0(a3, v22, 0);
    }
  }
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 40;
  *(_QWORD *)(a1 + 16) = a1 + 40;
  *(_QWORD *)(a1 + 64) = a1 + 96;
  *(_QWORD *)(a1 + 72) = a1 + 96;
  *(_QWORD *)(a1 + 24) = 0x100000002LL;
  *(_QWORD *)(a1 + 80) = 2;
  *(_QWORD *)(a1 + 40) = &unk_4F9EE48;
  *(_DWORD *)(a1 + 88) = 0;
  *(_DWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = 1;
  return a1;
}
