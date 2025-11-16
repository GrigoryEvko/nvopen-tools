// Function: sub_29DEA20
// Address: 0x29dea20
//
__int64 __fastcall sub_29DEA20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r13
  const char *v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  _BYTE *v10; // rax
  size_t v11; // r14
  _BYTE *v13; // rax

  v5 = (__int64)sub_CB72A0();
  v6 = sub_BD5D20(a3);
  v8 = *(_BYTE **)(v5 + 32);
  v9 = (unsigned __int8 *)v6;
  v10 = *(_BYTE **)(v5 + 24);
  v11 = v7;
  if ( v10 - v8 < v7 )
  {
    v5 = sub_CB6200(v5, v9, v7);
    v10 = *(_BYTE **)(v5 + 24);
    v8 = *(_BYTE **)(v5 + 32);
  }
  else if ( v7 )
  {
    memcpy(v8, v9, v7);
    v13 = *(_BYTE **)(v5 + 24);
    v8 = (_BYTE *)(v11 + *(_QWORD *)(v5 + 32));
    *(_QWORD *)(v5 + 32) = v8;
    if ( v8 != v13 )
      goto LABEL_4;
LABEL_7:
    sub_CB6200(v5, (unsigned __int8 *)"\n", 1u);
    goto LABEL_5;
  }
  if ( v8 == v10 )
    goto LABEL_7;
LABEL_4:
  *v8 = 10;
  ++*(_QWORD *)(v5 + 32);
LABEL_5:
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
