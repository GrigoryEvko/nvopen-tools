// Function: sub_A5C590
// Address: 0xa5c590
//
_BYTE *__fastcall sub_A5C590(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rdi
  void *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdi
  _WORD *v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdi
  _BYTE *result; // rax
  __int64 v15; // [rsp+8h] [rbp-48h] BYREF
  __int64 v16[8]; // [rsp+10h] [rbp-40h] BYREF

  v3 = a1[1];
  v4 = a1[4];
  v16[0] = (__int64)off_4979428;
  v5 = (__int64)(a1 + 5);
  v6 = *a1;
  v16[2] = v4;
  v16[3] = v3;
  v7 = *(void **)(v6 + 32);
  v8 = *(_QWORD *)(v6 + 24);
  v16[1] = v5;
  if ( (unsigned __int64)(v8 - (_QWORD)v7) <= 0xA )
  {
    sub_CB6200(v6, "#dbg_label(", 11);
  }
  else
  {
    qmemcpy(v7, "#dbg_label(", 11);
    *(_QWORD *)(v6 + 32) += 11LL;
  }
  sub_A5C090(*a1, *(_QWORD *)(a2 + 40), v16);
  v9 = *a1;
  v10 = *(_WORD **)(*a1 + 32);
  if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v10 <= 1u )
  {
    sub_CB6200(v9, ", ", 2);
  }
  else
  {
    *v10 = 8236;
    *(_QWORD *)(v9 + 32) += 2LL;
  }
  v11 = *(_QWORD *)(a2 + 24);
  v15 = v11;
  if ( v11 )
    sub_B96E90(&v15, v11, 1);
  v12 = sub_B10CD0(&v15);
  sub_A5C090(*a1, v12, v16);
  if ( v15 )
    sub_B91220(&v15);
  v13 = *a1;
  result = *(_BYTE **)(*a1 + 32);
  if ( *(_BYTE **)(*a1 + 24) == result )
    return (_BYTE *)sub_CB6200(v13, ")", 1);
  *result = 41;
  ++*(_QWORD *)(v13 + 32);
  return result;
}
