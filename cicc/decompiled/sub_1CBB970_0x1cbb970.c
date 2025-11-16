// Function: sub_1CBB970
// Address: 0x1cbb970
//
char __fastcall sub_1CBB970(_QWORD *a1, unsigned __int64 *a2)
{
  _QWORD *v3; // r12
  __int64 v4; // r8
  unsigned __int64 v5; // rdx
  _QWORD *v6; // r15
  _QWORD *v7; // rbx
  _QWORD *v8; // rax
  _QWORD *v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rcx
  _QWORD *v13; // rdi
  __int64 v14; // rax

  v3 = a1 + 1;
  v4 = a1[2];
  if ( v4 )
  {
    v5 = *a2;
    v6 = a1 + 1;
    v7 = (_QWORD *)a1[2];
    while ( 1 )
    {
      while ( v7[4] < v5 )
      {
        v7 = (_QWORD *)v7[3];
        if ( !v7 )
          goto LABEL_7;
      }
      v8 = (_QWORD *)v7[2];
      if ( v7[4] <= v5 )
        break;
      v6 = v7;
      v7 = (_QWORD *)v7[2];
      if ( !v8 )
      {
LABEL_7:
        LOBYTE(v8) = v3 == v6;
        goto LABEL_8;
      }
    }
    v9 = (_QWORD *)v7[3];
    if ( v9 )
    {
      do
      {
        while ( 1 )
        {
          v10 = v9[2];
          v11 = v9[3];
          if ( v9[4] > v5 )
            break;
          v9 = (_QWORD *)v9[3];
          if ( !v11 )
            goto LABEL_17;
        }
        v6 = v9;
        v9 = (_QWORD *)v9[2];
      }
      while ( v10 );
    }
LABEL_17:
    while ( v8 )
    {
      while ( 1 )
      {
        v12 = v8[3];
        if ( v8[4] >= v5 )
          break;
        v8 = (_QWORD *)v8[3];
        if ( !v12 )
          goto LABEL_20;
      }
      v7 = v8;
      v8 = (_QWORD *)v8[2];
    }
LABEL_20:
    if ( (_QWORD *)a1[3] != v7 )
      goto LABEL_24;
    if ( v6 != v3 )
    {
      while ( v6 != v7 )
      {
        v13 = v7;
        v7 = (_QWORD *)sub_220EF30(v7);
        v14 = sub_220F330(v13, v3);
        LOBYTE(v8) = j_j___libc_free_0(v14, 40);
        --a1[5];
LABEL_24:
        ;
      }
      return (char)v8;
    }
LABEL_10:
    LOBYTE(v8) = sub_1CBB7A0(v4);
    a1[2] = 0;
    a1[3] = v3;
    a1[4] = v3;
    a1[5] = 0;
    return (char)v8;
  }
  v6 = a1 + 1;
  LOBYTE(v8) = 1;
LABEL_8:
  if ( (_QWORD *)a1[3] == v6 && (_BYTE)v8 )
    goto LABEL_10;
  return (char)v8;
}
