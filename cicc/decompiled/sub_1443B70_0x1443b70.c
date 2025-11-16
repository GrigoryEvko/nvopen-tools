// Function: sub_1443B70
// Address: 0x1443b70
//
__int64 __fastcall sub_1443B70(__int64 a1, unsigned __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  _QWORD *v9; // r15
  _QWORD *v10; // rax
  _QWORD *v11; // rsi
  __int64 v12; // rdx
  _QWORD *v13; // r13
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rax
  char v18; // al
  __int64 v19; // r8
  __int64 v20; // rax
  _QWORD *v21; // rsi
  _QWORD *v22; // rax
  _QWORD *i; // r15
  unsigned __int64 v24; // rsi
  _QWORD *v25; // rax
  _QWORD *v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // r13
  char v29; // al
  _QWORD *v30; // [rsp+0h] [rbp-40h]
  __int64 v31; // [rsp+8h] [rbp-38h]
  __int64 v32; // [rsp+8h] [rbp-38h]

  v8 = *(_QWORD *)(a1 + 24);
  v9 = (_QWORD *)(v8 + 8);
  v10 = *(_QWORD **)(v8 + 16);
  if ( v10 )
  {
    v11 = v9;
    do
    {
      while ( 1 )
      {
        a4 = v10[2];
        v12 = v10[3];
        if ( v10[4] >= a2 )
          break;
        v10 = (_QWORD *)v10[3];
        if ( !v12 )
          goto LABEL_6;
      }
      v11 = v10;
      v10 = (_QWORD *)v10[2];
    }
    while ( a4 );
LABEL_6:
    if ( v9 != v11 && v11[4] <= a2 )
      v9 = v11;
  }
  v13 = v9 + 6;
  if ( (unsigned __int8)sub_15CC8F0(*(_QWORD *)(a1 + 8), a2, a3, a4, a5) )
  {
    v20 = *(_QWORD *)(a1 + 24);
    v21 = (_QWORD *)(v20 + 8);
    v22 = *(_QWORD **)(v20 + 16);
    v30 = v21;
    if ( v22 )
    {
      do
      {
        while ( 1 )
        {
          v14 = v22[2];
          v27 = v22[3];
          if ( v22[4] >= a3 )
            break;
          v22 = (_QWORD *)v22[3];
          if ( !v27 )
            goto LABEL_32;
        }
        v21 = v22;
        v22 = (_QWORD *)v22[2];
      }
      while ( v14 );
LABEL_32:
      if ( v30 != v21 )
      {
        if ( v21[4] > a3 )
          v21 = v30;
        v30 = v21;
      }
    }
    v19 = v9[8];
    for ( i = v30 + 6; (_QWORD *)v19 != v13; v19 = sub_220EF30(v19) )
    {
      v24 = *(_QWORD *)(v19 + 32);
      if ( v24 != a2 && v24 != a3 )
      {
        v25 = (_QWORD *)v30[7];
        if ( !v25 )
          return 0;
        v26 = v30 + 6;
        do
        {
          if ( v25[4] < v24 )
          {
            v25 = (_QWORD *)v25[3];
          }
          else
          {
            v26 = v25;
            v25 = (_QWORD *)v25[2];
          }
        }
        while ( v25 );
        if ( i == v26 )
          return 0;
        if ( v26[4] > v24 )
          return 0;
        v31 = v19;
        v18 = sub_1443AC0(a1, v24, a2, a3);
        v19 = v31;
        if ( !v18 )
          return 0;
      }
    }
    v28 = v30[8];
    if ( (_QWORD *)v28 != i )
    {
      while ( 1 )
      {
        v32 = *(_QWORD *)(v28 + 32);
        v29 = sub_15CC890(*(_QWORD *)(a1 + 8), a2, v32, v14, v19);
        if ( a3 != v32 )
        {
          if ( v29 )
            break;
        }
        v28 = sub_220EF30(v28);
        if ( (_QWORD *)v28 == i )
          return 1;
      }
      return 0;
    }
    return 1;
  }
  v15 = v9[8];
  if ( (_QWORD *)v15 == v13 )
    return 1;
  while ( 1 )
  {
    v16 = *(_QWORD *)(v15 + 32);
    if ( v16 != a2 && v16 != a3 )
      break;
    v15 = sub_220EF30(v15);
    if ( (_QWORD *)v15 == v13 )
      return 1;
  }
  return 0;
}
