// Function: sub_A5A2A0
// Address: 0xa5a2a0
//
void __fastcall sub_A5A2A0(__int64 a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // rbx
  __int64 v3; // rsi
  _QWORD *v4; // rbx
  _QWORD *v5; // rbx
  _QWORD *v6; // r15
  _QWORD *v7; // r14
  unsigned int v8; // ebx
  int v9; // r13d
  __int64 v10; // rsi
  _BYTE *v11; // rax
  _QWORD *v12; // rbx
  _QWORD *i; // r13
  __int64 v14; // rsi
  __int64 v15; // rdx
  char v16; // al
  char v17; // [rsp+Fh] [rbp-41h] BYREF
  __int64 v18; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = *(_QWORD **)(a1 + 8);
  v2 = (_QWORD *)v1[2];
  if ( v1 + 1 != v2 )
  {
    do
    {
      if ( !v2 )
        BUG();
      if ( (*((_BYTE *)v2 - 49) & 0x10) == 0 )
        sub_A58EF0(a1, (__int64)(v2 - 7));
      sub_A59D20(a1, (__int64)(v2 - 7));
      v3 = v2[2];
      if ( v3 )
        sub_A59250(a1, v3);
      v2 = (_QWORD *)v2[1];
    }
    while ( v1 + 1 != v2 );
    v1 = *(_QWORD **)(a1 + 8);
  }
  v4 = (_QWORD *)v1[6];
  if ( v4 != v1 + 5 )
  {
    do
    {
      if ( !v4 )
        BUG();
      if ( (*((_BYTE *)v4 - 41) & 0x10) == 0 )
        sub_A58EF0(a1, (__int64)(v4 - 6));
      v4 = (_QWORD *)v4[1];
    }
    while ( v1 + 5 != v4 );
    v1 = *(_QWORD **)(a1 + 8);
  }
  v5 = (_QWORD *)v1[8];
  if ( v1 + 7 != v5 )
  {
    do
    {
      if ( !v5 )
        BUG();
      if ( (*((_BYTE *)v5 - 49) & 0x10) == 0 )
        sub_A58EF0(a1, (__int64)(v5 - 7));
      v5 = (_QWORD *)v5[1];
    }
    while ( v1 + 7 != v5 );
    v1 = *(_QWORD **)(a1 + 8);
  }
  v6 = (_QWORD *)v1[10];
  v7 = v1 + 9;
  if ( v1 + 9 != v6 )
  {
    do
    {
      v8 = 0;
      v9 = sub_B91A00(v6);
      if ( v9 )
      {
        do
        {
          v10 = v8++;
          v11 = (_BYTE *)sub_B91A10(v6, v10);
          sub_A59AF0(a1, v11);
        }
        while ( v9 != v8 );
      }
      v6 = (_QWORD *)v6[1];
    }
    while ( v7 != v6 );
    v1 = *(_QWORD **)(a1 + 8);
  }
  v12 = (_QWORD *)v1[4];
  for ( i = v1 + 3; i != v12; v12 = (_QWORD *)v12[1] )
  {
    if ( !v12 )
      BUG();
    if ( (*((_BYTE *)v12 - 49) & 0x10) == 0 )
      sub_A58EF0(a1, (__int64)(v12 - 7));
    if ( *(_BYTE *)(a1 + 25) )
      sub_A59FE0(a1, (__int64)(v12 - 7));
    v19[0] = v12[8];
    v14 = sub_A74680(v19);
    if ( v14 )
      sub_A59250(a1, v14);
  }
  if ( *(_QWORD *)(a1 + 48) )
  {
    v15 = *(_QWORD *)(a1 + 8);
    v18 = a1;
    v16 = *(_BYTE *)(a1 + 25);
    v19[0] = v15;
    v17 = v16;
    (*(void (__fastcall **)(__int64, __int64 *, _QWORD *, char *))(a1 + 56))(a1 + 32, &v18, v19, &v17);
  }
}
