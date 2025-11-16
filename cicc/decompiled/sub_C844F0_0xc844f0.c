// Function: sub_C844F0
// Address: 0xc844f0
//
void __fastcall sub_C844F0(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // r13
  __int64 v12; // rax
  char *v13; // rdx
  char *v14; // rcx
  char v15; // si
  const void *v16; // rsi
  unsigned __int64 v17; // rdx
  const void *v18; // rsi

  if ( a1 == a2 )
    return;
  v3 = (_QWORD *)*a1;
  if ( a1 + 3 != (_QWORD *)*a1 && (_QWORD *)*a2 != a2 + 3 )
  {
    *a1 = *a2;
    v4 = a2[1];
    *a2 = v3;
    v5 = a1[1];
    a1[1] = v4;
    v6 = a2[2];
    a2[1] = v5;
    v7 = a1[2];
    a1[2] = v6;
    a2[2] = v7;
    return;
  }
  v8 = a2[1];
  if ( v8 > a1[2] )
  {
    sub_C8D290(a1, a1 + 3, v8, 1);
    v9 = a1[1];
    if ( v9 <= a2[2] )
      goto LABEL_8;
    goto LABEL_22;
  }
  v9 = a1[1];
  if ( v9 > a2[2] )
  {
LABEL_22:
    sub_C8D290(a2, a2 + 3, v9, 1);
    v9 = a1[1];
  }
LABEL_8:
  v10 = a2[1];
  v11 = v9;
  if ( v10 <= v9 )
    v11 = a2[1];
  if ( v11 )
  {
    v12 = 0;
    do
    {
      v13 = (char *)(v12 + *a2);
      v14 = (char *)(v12 + *a1);
      ++v12;
      v15 = *v14;
      *v14 = *v13;
      *v13 = v15;
    }
    while ( v12 != v11 );
    v9 = a1[1];
    v10 = a2[1];
  }
  if ( v9 <= v10 )
  {
    if ( v9 < v10 )
    {
      v17 = v9;
      v18 = (const void *)(*a2 + v11);
      if ( v18 != (const void *)(v10 + *a2) )
      {
        memcpy((void *)(v9 + *a1), v18, v10 - v11);
        v17 = a1[1];
      }
      a1[1] = v17 + v10 - v9;
      a2[1] = v11;
    }
  }
  else
  {
    v16 = (const void *)(*a1 + v11);
    if ( v16 != (const void *)(v9 + *a1) )
    {
      memcpy((void *)(v10 + *a2), v16, v9 - v11);
      v9 = a2[1] + v9 - v10;
    }
    a2[1] = v9;
    a1[1] = v11;
  }
}
