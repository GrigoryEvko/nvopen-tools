// Function: sub_27C08A0
// Address: 0x27c08a0
//
__int64 __fastcall sub_27C08A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // r15
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 *v9; // r10
  char v10; // al
  char v11; // al
  __int64 v12; // r14
  char v13; // al
  __int64 *v14; // r15
  char v16; // al
  __int64 v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h]

  v18 = (a3 - 1) / 2;
  if ( a2 >= v18 )
  {
    if ( (a3 & 1) != 0 )
    {
      v14 = (__int64 *)(a1 + 8 * a2);
      goto LABEL_18;
    }
    v7 = a2;
LABEL_20:
    if ( (a3 - 2) / 2 == v7 )
    {
      *(_QWORD *)(a1 + 8 * v7) = *(_QWORD *)(a1 + 8 * (2 * v7 + 2) - 8);
      v7 = 2 * v7 + 1;
    }
    goto LABEL_11;
  }
  for ( i = a2; ; i = v7 )
  {
    v7 = 2 * (i + 1);
    v9 = (__int64 *)(a1 + 16 * (i + 1));
    v8 = *(v9 - 1);
    if ( *v9 == v8 )
      goto LABEL_4;
    v19 = *v9;
    sub_B196A0(*(_QWORD *)(a5 + 16), *v9, v8);
    if ( v10 )
    {
      --v7;
      v8 = *(_QWORD *)(a1 + 8 * v7);
LABEL_4:
      *(_QWORD *)(a1 + 8 * i) = v8;
      if ( v7 >= v18 )
        break;
      continue;
    }
    sub_B196A0(*(_QWORD *)(a5 + 16), v8, v19);
    if ( !v11 )
      goto LABEL_26;
    *(_QWORD *)(a1 + 8 * i) = *(_QWORD *)(a1 + 16 * (i + 1));
    if ( v7 >= v18 )
      break;
  }
  if ( (a3 & 1) == 0 )
    goto LABEL_20;
LABEL_11:
  v12 = (v7 - 1) / 2;
  if ( v7 > a2 )
  {
    while ( 1 )
    {
      v14 = (__int64 *)(a1 + 8 * v12);
      if ( a4 == *v14 )
        goto LABEL_17;
      v20 = *v14;
      sub_B196A0(*(_QWORD *)(a5 + 16), *v14, a4);
      if ( !v13 )
        break;
      *(_QWORD *)(a1 + 8 * v7) = *v14;
      v7 = v12;
      if ( a2 >= v12 )
        goto LABEL_18;
      v12 = (v12 - 1) / 2;
    }
    sub_B196A0(*(_QWORD *)(a5 + 16), a4, v20);
    if ( !v16 )
LABEL_26:
      BUG();
  }
LABEL_17:
  v14 = (__int64 *)(a1 + 8 * v7);
LABEL_18:
  *v14 = a4;
  return a4;
}
