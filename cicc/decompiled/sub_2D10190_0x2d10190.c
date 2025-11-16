// Function: sub_2D10190
// Address: 0x2d10190
//
void __fastcall sub_2D10190(_QWORD *a1, unsigned __int64 *a2)
{
  int *v3; // r12
  unsigned __int64 v4; // r8
  unsigned __int64 v5; // rdx
  int *v6; // r15
  int *v7; // rbx
  int *v8; // rax
  bool v9; // al
  int *v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rcx
  int *v14; // rdi
  int *v15; // rax

  v3 = (int *)(a1 + 1);
  v4 = a1[2];
  if ( v4 )
  {
    v5 = *a2;
    v6 = (int *)(a1 + 1);
    v7 = (int *)a1[2];
    while ( 1 )
    {
      while ( *((_QWORD *)v7 + 4) < v5 )
      {
        v7 = (int *)*((_QWORD *)v7 + 3);
        if ( !v7 )
          goto LABEL_7;
      }
      v8 = (int *)*((_QWORD *)v7 + 2);
      if ( *((_QWORD *)v7 + 4) <= v5 )
        break;
      v6 = v7;
      v7 = (int *)*((_QWORD *)v7 + 2);
      if ( !v8 )
      {
LABEL_7:
        v9 = v3 == v6;
        goto LABEL_8;
      }
    }
    v10 = (int *)*((_QWORD *)v7 + 3);
    if ( v10 )
    {
      do
      {
        while ( 1 )
        {
          v11 = *((_QWORD *)v10 + 2);
          v12 = *((_QWORD *)v10 + 3);
          if ( *((_QWORD *)v10 + 4) > v5 )
            break;
          v10 = (int *)*((_QWORD *)v10 + 3);
          if ( !v12 )
            goto LABEL_17;
        }
        v6 = v10;
        v10 = (int *)*((_QWORD *)v10 + 2);
      }
      while ( v11 );
    }
LABEL_17:
    while ( v8 )
    {
      while ( 1 )
      {
        v13 = *((_QWORD *)v8 + 3);
        if ( *((_QWORD *)v8 + 4) >= v5 )
          break;
        v8 = (int *)*((_QWORD *)v8 + 3);
        if ( !v13 )
          goto LABEL_20;
      }
      v7 = v8;
      v8 = (int *)*((_QWORD *)v8 + 2);
    }
LABEL_20:
    if ( (int *)a1[3] != v7 )
      goto LABEL_24;
    if ( v6 != v3 )
    {
      while ( v6 != v7 )
      {
        v14 = v7;
        v7 = (int *)sub_220EF30((__int64)v7);
        v15 = sub_220F330(v14, v3);
        j_j___libc_free_0((unsigned __int64)v15);
        --a1[5];
LABEL_24:
        ;
      }
      return;
    }
LABEL_10:
    sub_2D0FFC0(v4);
    a1[2] = 0;
    a1[3] = v3;
    a1[4] = v3;
    a1[5] = 0;
    return;
  }
  v6 = (int *)(a1 + 1);
  v9 = 1;
LABEL_8:
  if ( (int *)a1[3] == v6 && v9 )
    goto LABEL_10;
}
