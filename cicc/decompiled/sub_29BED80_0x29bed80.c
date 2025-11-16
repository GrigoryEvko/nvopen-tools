// Function: sub_29BED80
// Address: 0x29bed80
//
_QWORD *__fastcall sub_29BED80(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v5; // rsi
  unsigned __int64 v6; // rax
  int v7; // edx
  __int64 v8; // rbx
  unsigned __int64 v9; // rax
  _QWORD *v10; // r12
  __int64 v12; // r14
  _QWORD *result; // rax
  _QWORD *i; // r13
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // [rsp+0h] [rbp-50h]
  __int64 v19; // [rsp+8h] [rbp-48h]

  v5 = (_QWORD *)(a2 + 48);
  v6 = *v5 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v6 == v5 )
  {
    v8 = 0;
  }
  else
  {
    if ( !v6 )
      BUG();
    v7 = *(unsigned __int8 *)(v6 - 24);
    v8 = 0;
    v9 = v6 - 24;
    if ( (unsigned int)(v7 - 30) < 0xB )
      v8 = v9;
  }
  v10 = a1 + 6;
  v19 = v8 + 24;
  v12 = v8;
LABEL_6:
  result = a1;
  for ( i = (_QWORD *)a1[7]; i != v10; i = (_QWORD *)a1[7] )
  {
    result = i;
    v16 = 0;
    do
    {
      result = (_QWORD *)result[1];
      ++v16;
    }
    while ( v10 != result );
    if ( v16 == 1 )
      break;
    if ( i )
      i -= 3;
    if ( !(unsigned __int8)sub_29BDD80(i, v12, a3, a4, a5, 0) )
      goto LABEL_6;
    v17 = v18;
    LOWORD(v17) = 0;
    v18 = v17;
    sub_B44500(i, v19, v17);
    result = a1;
  }
  return result;
}
