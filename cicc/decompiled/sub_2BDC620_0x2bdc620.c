// Function: sub_2BDC620
// Address: 0x2bdc620
//
__int64 __fastcall sub_2BDC620(__int64 a1, __int64 a2)
{
  _BYTE *v4; // rax
  unsigned __int8 *v5; // rsi
  size_t v6; // r14
  _BYTE *v7; // rdi
  __int64 v8; // r15
  __int64 *v9; // rbx
  __int64 result; // rax
  __int64 *i; // r12
  __int64 v12; // rdi
  _BYTE *v13; // rax

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
      v13 = *(_BYTE **)(a2 + 24);
      v7 = (_BYTE *)(v6 + *(_QWORD *)(a2 + 32));
      *(_QWORD *)(a2 + 32) = v7;
      if ( v7 != v13 )
        goto LABEL_4;
      goto LABEL_9;
    }
  }
  if ( v7 != v4 )
  {
LABEL_4:
    *v7 = 10;
    ++*(_QWORD *)(v8 + 32);
    goto LABEL_5;
  }
LABEL_9:
  sub_CB6200(v8, (unsigned __int8 *)"\n", 1u);
LABEL_5:
  v9 = *(__int64 **)(a1 + 40);
  result = *(unsigned int *)(a1 + 48);
  for ( i = &v9[result]; i != v9; result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v12 + 16LL))(
                                             v12,
                                             a2) )
    v12 = *v9++;
  return result;
}
