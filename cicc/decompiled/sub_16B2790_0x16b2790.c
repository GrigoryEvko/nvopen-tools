// Function: sub_16B2790
// Address: 0x16b2790
//
__int64 __fastcall sub_16B2790(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r13d
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r12
  void *v9; // rdi
  unsigned __int64 v10; // r14
  const char *v11; // rsi
  __int64 v13; // rax

  v4 = a2;
  v6 = sub_16E8C20(a1, a2, a3, a4);
  v7 = *(_QWORD *)(v6 + 24);
  v8 = v6;
  if ( (unsigned __int64)(*(_QWORD *)(v6 + 16) - v7) <= 2 )
  {
    v13 = sub_16E7EE0(v6, "  -", 3);
    v9 = *(void **)(v13 + 24);
    v8 = v13;
  }
  else
  {
    *(_BYTE *)(v7 + 2) = 45;
    *(_WORD *)v7 = 8224;
    v9 = (void *)(*(_QWORD *)(v6 + 24) + 3LL);
    *(_QWORD *)(v6 + 24) = v9;
  }
  v10 = *(_QWORD *)(a1 + 32);
  v11 = *(const char **)(a1 + 24);
  if ( *(_QWORD *)(v8 + 16) - (_QWORD)v9 < v10 )
  {
    sub_16E7EE0(v8, v11, *(_QWORD *)(a1 + 32));
    v10 = *(_QWORD *)(a1 + 32);
  }
  else if ( v10 )
  {
    memcpy(v9, v11, *(_QWORD *)(a1 + 32));
    *(_QWORD *)(v8 + 24) += v10;
    v10 = *(_QWORD *)(a1 + 32);
  }
  return sub_16B2520(*(_OWORD *)(a1 + 40), v4, (int)v10 + 6);
}
