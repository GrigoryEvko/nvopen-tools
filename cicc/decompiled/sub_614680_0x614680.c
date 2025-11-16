// Function: sub_614680
// Address: 0x614680
//
_QWORD *sub_614680()
{
  _QWORD *v0; // rbx
  _QWORD *v1; // rax
  _QWORD *v2; // rcx
  _QWORD *v3; // rdx
  __int64 *v4; // rax
  __int64 *v5; // rbx
  __int64 v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v8; // rax
  _QWORD *v9; // rcx
  _QWORD *v10; // rdx
  _QWORD *v11; // rbx
  _QWORD *v12; // rax
  _QWORD *v13; // rcx
  _QWORD *v14; // rdx
  _QWORD *v15; // rbx
  _QWORD *v16; // rax
  _QWORD *v17; // rcx
  _QWORD *v18; // rdx

  v0 = (_QWORD *)sub_822B10(16);
  if ( v0 )
  {
    v1 = (_QWORD *)sub_822B10(256);
    v2 = v1;
    v3 = v1 + 32;
    do
    {
      if ( v1 )
        *v1 = 0;
      v1 += 2;
    }
    while ( v1 != v3 );
    *v0 = v2;
    v0[1] = 15;
  }
  unk_4D048F0 = v0;
  v4 = (__int64 *)sub_822B10(24);
  v5 = v4;
  if ( v4 )
  {
    *v4 = 0;
    v4[1] = 0;
    v4[2] = 0;
    v6 = sub_822B10(32);
    v5[1] = 4;
    *v5 = v6;
  }
  unk_4D048E8 = v5;
  v7 = (_QWORD *)sub_822B10(16);
  if ( v7 )
  {
    v8 = (_QWORD *)sub_822B10(256);
    v9 = v8;
    v10 = v8 + 32;
    do
    {
      if ( v8 )
        *v8 = 0;
      v8 += 2;
    }
    while ( v8 != v10 );
    *v7 = v9;
    v7[1] = 15;
  }
  unk_4D048E0 = v7;
  v11 = (_QWORD *)sub_822B10(16);
  if ( v11 )
  {
    v12 = (_QWORD *)sub_822B10(256);
    v13 = v12;
    v14 = v12 + 32;
    do
    {
      if ( v12 )
        *v12 = 0;
      v12 += 2;
    }
    while ( v14 != v12 );
    *v11 = v13;
    v11[1] = 15;
  }
  unk_4D048D8 = v11;
  v15 = (_QWORD *)sub_822B10(16);
  if ( v15 )
  {
    v16 = (_QWORD *)sub_822B10(256);
    v17 = v16;
    v18 = v16 + 32;
    do
    {
      if ( v16 )
        *v16 = 0;
      v16 += 2;
    }
    while ( v16 != v18 );
    *v15 = v17;
    v15[1] = 15;
  }
  qword_4D048D0 = v15;
  return &qword_4D048D0;
}
