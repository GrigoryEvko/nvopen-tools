// Function: sub_1E62FF0
// Address: 0x1e62ff0
//
__int64 __fastcall sub_1E62FF0(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  __int64 v6; // rax
  _QWORD *v7; // r15
  _QWORD *v8; // rax
  _QWORD *v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rdi
  _QWORD *v14; // r13
  __int64 v15; // rdi
  __int64 v16; // rax
  bool v18; // al
  __int64 v19; // r8
  __int64 v20; // rax
  _QWORD *v21; // rsi
  _QWORD *v22; // rax
  _QWORD *i; // r15
  unsigned __int64 v24; // rsi
  _QWORD *v25; // rax
  _QWORD *v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // r8
  __int64 v30; // r13
  bool v31; // al
  _QWORD *v32; // [rsp+0h] [rbp-40h]
  __int64 v33; // [rsp+0h] [rbp-40h]
  __int64 v34; // [rsp+8h] [rbp-38h]
  __int64 v35; // [rsp+8h] [rbp-38h]

  v6 = *(_QWORD *)(a1 + 24);
  v7 = (_QWORD *)(v6 + 240);
  v8 = *(_QWORD **)(v6 + 248);
  if ( v8 )
  {
    v9 = v7;
    do
    {
      while ( 1 )
      {
        v10 = v8[2];
        v11 = v8[3];
        if ( v8[4] >= a2 )
          break;
        v8 = (_QWORD *)v8[3];
        if ( !v11 )
          goto LABEL_6;
      }
      v9 = v8;
      v8 = (_QWORD *)v8[2];
    }
    while ( v10 );
LABEL_6:
    if ( v7 != v9 && v9[4] <= a2 )
      v7 = v9;
  }
  v12 = *(_QWORD *)(a1 + 8);
  sub_1E06620(v12);
  v13 = *(_QWORD *)(v12 + 1312);
  v14 = v7 + 6;
  if ( sub_1E05550(v13, a2, a3) )
  {
    v20 = *(_QWORD *)(a1 + 24);
    v21 = (_QWORD *)(v20 + 240);
    v22 = *(_QWORD **)(v20 + 248);
    v32 = v21;
    if ( v22 )
    {
      do
      {
        while ( 1 )
        {
          v27 = v22[2];
          v28 = v22[3];
          if ( v22[4] >= a3 )
            break;
          v22 = (_QWORD *)v22[3];
          if ( !v28 )
            goto LABEL_32;
        }
        v21 = v22;
        v22 = (_QWORD *)v22[2];
      }
      while ( v27 );
LABEL_32:
      if ( v32 != v21 )
      {
        if ( v21[4] > a3 )
          v21 = v32;
        v32 = v21;
      }
    }
    v19 = v7[8];
    for ( i = v32 + 6; (_QWORD *)v19 != v14; v19 = sub_220EF30(v19) )
    {
      v24 = *(_QWORD *)(v19 + 32);
      if ( v24 != a2 && v24 != a3 )
      {
        v25 = (_QWORD *)v32[7];
        if ( !v25 )
          return 0;
        v26 = v32 + 6;
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
        if ( v26 == i )
          return 0;
        if ( v26[4] > v24 )
          return 0;
        v34 = v19;
        v18 = sub_1E62F30(a1, v24, a2, a3);
        v19 = v34;
        if ( !v18 )
          return 0;
      }
    }
    v29 = v32[8];
    if ( (_QWORD *)v29 != i )
    {
      while ( 1 )
      {
        v30 = *(_QWORD *)(v29 + 32);
        v33 = v29;
        v35 = *(_QWORD *)(a1 + 8);
        sub_1E06620(v35);
        v31 = sub_1E054F0(*(_QWORD *)(v35 + 1312), a2, v30);
        if ( a3 != v30 && v31 )
          break;
        v29 = sub_220EF30(v33);
        if ( (_QWORD *)v29 == i )
          return 1;
      }
      return 0;
    }
    return 1;
  }
  v15 = v7[8];
  if ( (_QWORD *)v15 == v14 )
    return 1;
  while ( 1 )
  {
    v16 = *(_QWORD *)(v15 + 32);
    if ( v16 != a2 && v16 != a3 )
      break;
    v15 = sub_220EF30(v15);
    if ( (_QWORD *)v15 == v14 )
      return 1;
  }
  return 0;
}
