// Function: sub_16B2F80
// Address: 0x16b2f80
//
__int64 __fastcall sub_16B2F80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // ebx
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  _QWORD *v9; // r12
  void *v10; // rdi
  unsigned __int64 v11; // r14
  const char *v12; // rsi
  __int64 v13; // rax
  __int64 v15; // rax

  v5 = a3;
  v6 = sub_16E8C20(a1, a2, a3, a4);
  v8 = *(_QWORD *)(v6 + 24);
  v9 = (_QWORD *)v6;
  if ( (unsigned __int64)(*(_QWORD *)(v6 + 16) - v8) <= 2 )
  {
    v15 = sub_16E7EE0(v6, "  -", 3);
    v10 = *(void **)(v15 + 24);
    v9 = (_QWORD *)v15;
  }
  else
  {
    *(_BYTE *)(v8 + 2) = 45;
    *(_WORD *)v8 = 8224;
    v10 = (void *)(*(_QWORD *)(v6 + 24) + 3LL);
    *(_QWORD *)(v6 + 24) = v10;
  }
  v11 = *(_QWORD *)(a2 + 32);
  v12 = *(const char **)(a2 + 24);
  if ( v9[2] - (_QWORD)v10 < v11 )
  {
    v10 = v9;
    sub_16E7EE0(v9, v12, *(_QWORD *)(a2 + 32));
  }
  else if ( v11 )
  {
    memcpy(v10, v12, *(_QWORD *)(a2 + 32));
    v9[3] += v11;
  }
  v13 = sub_16E8C20(v10, v12, v8, v7);
  return sub_16E8750(v13, (unsigned int)(v5 - *(_DWORD *)(a2 + 32)));
}
