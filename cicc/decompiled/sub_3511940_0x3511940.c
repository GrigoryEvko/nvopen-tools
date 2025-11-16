// Function: sub_3511940
// Address: 0x3511940
//
__int64 *__fastcall sub_3511940(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 *a5)
{
  __int64 v5; // rsi
  __int64 v6; // rbx
  __int64 v9; // r14
  __int64 *v10; // r12
  __int64 *v13; // [rsp+8h] [rbp-48h]
  __int64 v14; // [rsp+10h] [rbp-40h]
  unsigned int v15; // [rsp+1Ch] [rbp-34h]

  v5 = a2 - (_QWORD)a1;
  v6 = v5 >> 3;
  if ( v5 <= 0 )
    return a1;
  v13 = a1;
  do
  {
    while ( 1 )
    {
      v9 = v6 >> 1;
      v10 = &v13[v6 >> 1];
      v14 = *a3;
      v15 = sub_2E441D0(*(_QWORD *)(a4 + 528), *a5, *v10);
      if ( v15 < (unsigned int)sub_2E441D0(*(_QWORD *)(a4 + 528), *a5, v14) )
        break;
      v6 = v6 - v9 - 1;
      v13 = v10 + 1;
      if ( v6 <= 0 )
        return v13;
    }
    v6 >>= 1;
  }
  while ( v9 > 0 );
  return v13;
}
