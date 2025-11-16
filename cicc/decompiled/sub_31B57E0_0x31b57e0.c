// Function: sub_31B57E0
// Address: 0x31b57e0
//
__int64 __fastcall sub_31B57E0(__int64 a1, __int64 a2)
{
  _BYTE *v4; // rax
  unsigned __int8 *v5; // rsi
  size_t v6; // r14
  _BYTE *v7; // rdi
  __int64 v8; // r15
  _BYTE *v9; // rax
  _BYTE *v10; // rdi
  unsigned __int64 v11; // r14
  unsigned __int8 *v12; // rsi
  __int64 v13; // r15
  __int64 *v14; // rbx
  __int64 result; // rax
  __int64 *i; // r13
  __int64 v17; // rdi
  _BYTE *v18; // rax
  _BYTE *v19; // rax

  v4 = *(_BYTE **)(a2 + 24);
  v5 = *(unsigned __int8 **)(a1 + 8);
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(_BYTE **)(a2 + 32);
  if ( v6 > v4 - v7 )
  {
    v8 = sub_CB6200(a2, v5, v6);
    v4 = *(_BYTE **)(v8 + 24);
    v7 = *(_BYTE **)(v8 + 32);
  }
  else
  {
    v8 = a2;
    if ( v6 )
    {
      memcpy(v7, v5, v6);
      v19 = *(_BYTE **)(a2 + 24);
      v7 = (_BYTE *)(v6 + *(_QWORD *)(a2 + 32));
      *(_QWORD *)(a2 + 32) = v7;
      if ( v7 != v19 )
        goto LABEL_4;
      goto LABEL_15;
    }
  }
  if ( v7 != v4 )
  {
LABEL_4:
    *v7 = 10;
    ++*(_QWORD *)(v8 + 32);
    goto LABEL_5;
  }
LABEL_15:
  sub_CB6200(v8, (unsigned __int8 *)"\n", 1u);
LABEL_5:
  v9 = *(_BYTE **)(a2 + 24);
  v10 = *(_BYTE **)(a2 + 32);
  v11 = *(_QWORD *)(a1 + 56);
  v12 = *(unsigned __int8 **)(a1 + 48);
  if ( v11 > v9 - v10 )
  {
    v13 = sub_CB6200(a2, v12, *(_QWORD *)(a1 + 56));
    v9 = *(_BYTE **)(v13 + 24);
    v10 = *(_BYTE **)(v13 + 32);
  }
  else
  {
    v13 = a2;
    if ( v11 )
    {
      memcpy(v10, v12, *(_QWORD *)(a1 + 56));
      v18 = *(_BYTE **)(a2 + 24);
      v10 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
      *(_QWORD *)(a2 + 32) = v10;
      if ( v18 != v10 )
        goto LABEL_8;
      goto LABEL_13;
    }
  }
  if ( v9 != v10 )
  {
LABEL_8:
    *v10 = 10;
    ++*(_QWORD *)(v13 + 32);
    goto LABEL_9;
  }
LABEL_13:
  sub_CB6200(v13, (unsigned __int8 *)"\n", 1u);
LABEL_9:
  v14 = *(__int64 **)(a1 + 80);
  result = *(unsigned int *)(a1 + 88);
  for ( i = &v14[result]; i != v14; result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v17 + 16LL))(
                                               v17,
                                               a2) )
    v17 = *v14++;
  return result;
}
