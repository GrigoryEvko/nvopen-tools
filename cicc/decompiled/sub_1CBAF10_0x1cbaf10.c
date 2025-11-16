// Function: sub_1CBAF10
// Address: 0x1cbaf10
//
char __fastcall sub_1CBAF10(_QWORD *a1, unsigned int *a2)
{
  _QWORD *v3; // r12
  __int64 v4; // r8
  unsigned int v5; // edx
  _QWORD *v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // rdi
  __int64 v14; // rax

  v3 = a1 + 1;
  v4 = a1[2];
  if ( v4 )
  {
    v5 = *a2;
    v6 = a1 + 1;
    v7 = a1[2];
    while ( 1 )
    {
      while ( *(_DWORD *)(v7 + 32) < v5 )
      {
        v7 = *(_QWORD *)(v7 + 24);
        if ( !v7 )
          goto LABEL_7;
      }
      v8 = *(_QWORD *)(v7 + 16);
      if ( *(_DWORD *)(v7 + 32) <= v5 )
        break;
      v6 = (_QWORD *)v7;
      v7 = *(_QWORD *)(v7 + 16);
      if ( !v8 )
      {
LABEL_7:
        LOBYTE(v8) = v3 == v6;
        goto LABEL_8;
      }
    }
    v9 = *(_QWORD *)(v7 + 24);
    if ( v9 )
    {
      do
      {
        while ( 1 )
        {
          v10 = *(_QWORD *)(v9 + 16);
          v11 = *(_QWORD *)(v9 + 24);
          if ( v5 < *(_DWORD *)(v9 + 32) )
            break;
          v9 = *(_QWORD *)(v9 + 24);
          if ( !v11 )
            goto LABEL_17;
        }
        v6 = (_QWORD *)v9;
        v9 = *(_QWORD *)(v9 + 16);
      }
      while ( v10 );
    }
LABEL_17:
    while ( v8 )
    {
      while ( 1 )
      {
        v12 = *(_QWORD *)(v8 + 24);
        if ( v5 <= *(_DWORD *)(v8 + 32) )
          break;
        v8 = *(_QWORD *)(v8 + 24);
        if ( !v12 )
          goto LABEL_20;
      }
      v7 = v8;
      v8 = *(_QWORD *)(v8 + 16);
    }
LABEL_20:
    if ( a1[3] != v7 )
      goto LABEL_24;
    if ( v6 != v3 )
    {
      while ( v6 != (_QWORD *)v7 )
      {
        v13 = v7;
        v7 = sub_220EF30(v7);
        v14 = sub_220F330(v13, v3);
        LOBYTE(v8) = j_j___libc_free_0(v14, 48);
        --a1[5];
LABEL_24:
        ;
      }
      return v8;
    }
LABEL_10:
    LOBYTE(v8) = sub_1CBAD40(v4);
    a1[2] = 0;
    a1[3] = v3;
    a1[4] = v3;
    a1[5] = 0;
    return v8;
  }
  v6 = a1 + 1;
  LOBYTE(v8) = 1;
LABEL_8:
  if ( (_QWORD *)a1[3] == v6 && (_BYTE)v8 )
    goto LABEL_10;
  return v8;
}
