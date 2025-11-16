// Function: sub_2D0F730
// Address: 0x2d0f730
//
void __fastcall sub_2D0F730(_QWORD *a1, unsigned int *a2)
{
  _QWORD *v3; // r12
  unsigned __int64 v4; // r8
  unsigned int v5; // edx
  _QWORD *v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rax
  bool v9; // al
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rcx
  int *v14; // rdi
  int *v15; // rax

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
        v9 = v3 == v6;
        goto LABEL_8;
      }
    }
    v10 = *(_QWORD *)(v7 + 24);
    if ( v10 )
    {
      do
      {
        while ( 1 )
        {
          v11 = *(_QWORD *)(v10 + 16);
          v12 = *(_QWORD *)(v10 + 24);
          if ( v5 < *(_DWORD *)(v10 + 32) )
            break;
          v10 = *(_QWORD *)(v10 + 24);
          if ( !v12 )
            goto LABEL_17;
        }
        v6 = (_QWORD *)v10;
        v10 = *(_QWORD *)(v10 + 16);
      }
      while ( v11 );
    }
LABEL_17:
    while ( v8 )
    {
      while ( 1 )
      {
        v13 = *(_QWORD *)(v8 + 24);
        if ( v5 <= *(_DWORD *)(v8 + 32) )
          break;
        v8 = *(_QWORD *)(v8 + 24);
        if ( !v13 )
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
        v14 = (int *)v7;
        v7 = sub_220EF30(v7);
        v15 = sub_220F330(v14, v3);
        j_j___libc_free_0((unsigned __int64)v15);
        --a1[5];
LABEL_24:
        ;
      }
      return;
    }
LABEL_10:
    sub_2D0F560(v4);
    a1[2] = 0;
    a1[3] = v3;
    a1[4] = v3;
    a1[5] = 0;
    return;
  }
  v6 = a1 + 1;
  v9 = 1;
LABEL_8:
  if ( (_QWORD *)a1[3] == v6 && v9 )
    goto LABEL_10;
}
