// Function: sub_8CC330
// Address: 0x8cc330
//
__int64 *__fastcall sub_8CC330(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rax
  _UNKNOWN *__ptr32 *v3; // r8
  __int64 v4; // rsi
  __int64 v5; // rcx
  __int64 v6; // rbx
  _QWORD *v7; // r15
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 *result; // rax
  _QWORD *v15; // rdx
  __int64 v16; // [rsp+0h] [rbp-40h]
  __int64 v17; // [rsp+8h] [rbp-38h]

  v1 = *(_QWORD *)(a1 + 96);
  v2 = sub_8C9880(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v1 + 32) + 88LL) + 104LL));
  v4 = *(_QWORD *)(v1 + 24);
  v5 = *(_QWORD *)(a1 + 88);
  v6 = *(_QWORD *)(v4 + 88);
  v17 = v5;
  v16 = *(_QWORD *)(*(_QWORD *)v2 + 88LL);
  v7 = *(_QWORD **)(v16 + 112);
  v8 = **(_QWORD **)(v6 + 216);
  if ( !v7 )
    return (__int64 *)sub_8CA1D0(v16, v4);
  while ( 1 )
  {
    v9 = *(_QWORD *)(v6 + 120);
    v10 = *(_QWORD *)(v7[1] + 88LL);
    v11 = *(_QWORD *)(v10 + 120);
    if ( v9 != v11 && !(unsigned int)sub_8D97D0(v9, v11, 0, v5, v3)
      || !sub_89AB40(**(_QWORD **)(v10 + 216), v8, 2, v5, v3) )
    {
      goto LABEL_4;
    }
    v12 = *(_QWORD *)(*(_QWORD *)(v6 + 216) + 8LL);
    v13 = *(_QWORD *)(*(_QWORD *)(v10 + 216) + 8LL);
    if ( !v12 )
      break;
    if ( sub_89AB40(v12, v13, 2, v5, v3) )
      goto LABEL_10;
LABEL_4:
    v7 = (_QWORD *)*v7;
    if ( !v7 )
    {
      v4 = *(_QWORD *)(v1 + 24);
      return (__int64 *)sub_8CA1D0(v16, v4);
    }
  }
  if ( v13 )
    goto LABEL_4;
LABEL_10:
  result = (__int64 *)v7[1];
  v15 = (_QWORD *)result[11];
  if ( v15 != (_QWORD *)v17 )
  {
    result = (__int64 *)v15[4];
    if ( !result )
      return sub_8CBB20(7u, v17, v15);
    v15 = (_QWORD *)*result;
    if ( v17 != *result )
      return sub_8CBB20(7u, v17, v15);
  }
  return result;
}
