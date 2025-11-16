// Function: sub_28942B0
// Address: 0x28942b0
//
__int64 __fastcall sub_28942B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 i; // r14
  __int64 v7; // r15
  __int64 *v8; // rbx
  char v9; // al
  __int64 v10; // r12
  __int64 *v11; // r14
  __int64 v13; // rax
  __int64 v15; // [rsp+10h] [rbp-50h]
  __int64 v17; // [rsp+28h] [rbp-38h]
  __int64 v18; // [rsp+28h] [rbp-38h]

  v5 = (a3 - 1) / 2;
  v15 = a3 & 1;
  if ( a2 >= v5 )
  {
    v7 = a2;
    v8 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    goto LABEL_15;
  }
  for ( i = a2; ; i = v7 )
  {
    v17 = a5;
    v7 = 2 * (i + 1);
    v8 = (__int64 *)(a1 + 16 * (i + 1));
    v9 = sub_B19DB0(*(_QWORD *)(a5 + 40), *v8, *(v8 - 1));
    a5 = v17;
    if ( v9 )
    {
      --v7;
      v8 = (__int64 *)(a1 + 8 * v7);
    }
    *(_QWORD *)(a1 + 8 * i) = *v8;
    if ( v7 >= v5 )
      break;
  }
  if ( !v15 )
  {
LABEL_15:
    if ( (a3 - 2) / 2 == v7 )
    {
      v13 = *(_QWORD *)(a1 + 8 * (2 * v7 + 2) - 8);
      v7 = 2 * v7 + 1;
      *v8 = v13;
      v8 = (__int64 *)(a1 + 8 * v7);
    }
  }
  v10 = (v7 - 1) / 2;
  if ( v7 > a2 )
  {
    while ( 1 )
    {
      v11 = (__int64 *)(a1 + 8 * v10);
      v18 = a5;
      v8 = (__int64 *)(a1 + 8 * v7);
      if ( !(unsigned __int8)sub_B19DB0(*(_QWORD *)(a5 + 40), *v11, a4) )
        break;
      a5 = v18;
      v7 = v10;
      *v8 = *v11;
      if ( a2 >= v10 )
      {
        v8 = (__int64 *)(a1 + 8 * v10);
        break;
      }
      v10 = (v10 - 1) / 2;
    }
  }
LABEL_13:
  *v8 = a4;
  return a4;
}
