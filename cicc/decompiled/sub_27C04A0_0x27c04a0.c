// Function: sub_27C04A0
// Address: 0x27c04a0
//
__int64 __fastcall sub_27C04A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 i; // r14
  __int64 v7; // r15
  __int64 *v8; // r13
  __int64 v9; // rdx
  char v10; // al
  __int64 v11; // rbx
  __int64 *v12; // r14
  char v13; // al
  __int64 v15; // rax
  __int64 v17; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+28h] [rbp-38h]
  __int64 v20; // [rsp+28h] [rbp-38h]

  v5 = (a3 - 1) / 2;
  v17 = a3 & 1;
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
    v7 = 2 * (i + 1);
    v8 = (__int64 *)(a1 + 16 * (i + 1));
    v9 = *(v8 - 1);
    if ( *v8 != v9 )
    {
      v19 = a5;
      sub_B196A0(*(_QWORD *)(a5 + 16), *v8, v9);
      a5 = v19;
      if ( v10 )
      {
        --v7;
        v8 = (__int64 *)(a1 + 8 * v7);
      }
      v9 = *v8;
    }
    *(_QWORD *)(a1 + 8 * i) = v9;
    if ( v7 >= v5 )
      break;
  }
  if ( !v17 )
  {
LABEL_15:
    if ( (a3 - 2) / 2 == v7 )
    {
      v15 = *(_QWORD *)(a1 + 8 * (2 * v7 + 2) - 8);
      v7 = 2 * v7 + 1;
      *v8 = v15;
      v8 = (__int64 *)(a1 + 8 * v7);
    }
  }
  v11 = (v7 - 1) / 2;
  if ( v7 > a2 )
  {
    while ( 1 )
    {
      v12 = (__int64 *)(a1 + 8 * v11);
      v8 = (__int64 *)(a1 + 8 * v7);
      if ( a4 == *v12 )
        break;
      v20 = a5;
      sub_B196A0(*(_QWORD *)(a5 + 16), *v12, a4);
      if ( !v13 )
        break;
      a5 = v20;
      v7 = v11;
      *v8 = *v12;
      if ( a2 >= v11 )
      {
        v8 = (__int64 *)(a1 + 8 * v11);
        break;
      }
      v11 = (v11 - 1) / 2;
    }
  }
LABEL_13:
  *v8 = a4;
  return a4;
}
