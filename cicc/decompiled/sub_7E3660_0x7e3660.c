// Function: sub_7E3660
// Address: 0x7e3660
//
__int64 *__fastcall sub_7E3660(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 *a5, _WORD *a6, _WORD *a7)
{
  __int64 v8; // r14
  char v9; // al
  __int64 v10; // rax
  __int64 v11; // rdx
  _QWORD *i; // rbx
  __int64 v13; // rax
  _QWORD *v14; // r13
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rcx
  __int16 v18; // ax
  __m128i *v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 *v23; // rax
  _QWORD *v25; // rax
  __int64 v30; // [rsp+28h] [rbp-38h]
  __int64 v31; // [rsp+28h] [rbp-38h]
  __int64 v32; // [rsp+28h] [rbp-38h]

  if ( a2 )
    a1 = *(_QWORD *)(a2 + 40);
  if ( !a3 )
  {
LABEL_52:
    v8 = 0;
    goto LABEL_21;
  }
  v8 = a3;
  v9 = *(_BYTE *)(a3 + 96);
  if ( (v9 & 8) == 0 )
    goto LABEL_20;
  while ( 1 )
  {
    if ( (v9 & 2) == 0 || !a2 )
    {
      v25 = *(_QWORD **)(*(_QWORD *)(v8 + 56) + 168LL);
      if ( v25[3] == v8 )
        goto LABEL_52;
      do
        v25 = (_QWORD *)*v25;
      while ( v25[3] != v8 );
      v8 = (__int64)v25;
      goto LABEL_49;
    }
    v10 = sub_8E5650(v8);
    if ( (*(_BYTE *)(v10 + 96) & 8) == 0 )
      goto LABEL_20;
    v11 = *(_QWORD *)(*(_QWORD *)(v10 + 56) + 168LL);
    if ( v10 == *(_QWORD *)(v11 + 24) )
      goto LABEL_20;
    for ( i = *(_QWORD **)v11; v10 != i[3]; i = (_QWORD *)*i )
      ;
    v13 = i[5];
    if ( v13 == *(_QWORD *)(a2 + 40) )
      break;
    v14 = **(_QWORD ***)(*(_QWORD *)(v8 + 56) + 168LL);
    if ( !v14 )
      goto LABEL_20;
    while ( v14[5] != v13 || i != (_QWORD *)sub_8E5650(v14) )
    {
      v14 = (_QWORD *)*v14;
      if ( !v14 )
        goto LABEL_20;
      v13 = i[5];
    }
    v8 = (__int64)v14;
LABEL_49:
    v9 = *(_BYTE *)(v8 + 96);
    if ( (v9 & 8) == 0 )
      goto LABEL_20;
  }
  if ( (_QWORD *)a2 == i )
    goto LABEL_52;
LABEL_20:
  while ( (*(_BYTE *)(*(_QWORD *)(v8 + 40) + 176LL) & 0x50) == 0 )
    v8 = *(_QWORD *)(v8 + 24);
LABEL_21:
  v15 = *(_QWORD *)(a1 + 168);
  v16 = sub_823970(48);
  *(_QWORD *)v16 = 0;
  v17 = v16;
  *(_QWORD *)(v16 + 8) = 0;
  *(_QWORD *)(v16 + 24) = 0;
  *(_QWORD *)(v16 + 32) = 0;
  *(_BYTE *)(v16 + 40) = 1;
  *(_QWORD *)(v16 + 16) = a2;
  v18 = *a6 + 1;
  *a6 = v18;
  if ( !*a7 )
    *a7 = v18;
  if ( a2 || !a3 )
  {
    if ( v8 )
    {
      *(_QWORD *)(v17 + 8) = v8;
      if ( !a2 )
      {
LABEL_27:
        v30 = v17;
        v19 = *(__m128i **)(v15 + 192);
        v20 = sub_7E3470(a1, v8);
        v17 = v30;
        v21 = *(_QWORD *)(v8 + 128) + v20;
        goto LABEL_43;
      }
    }
    else
    {
      *(_BYTE *)(v17 + 40) = 0;
      *(_QWORD *)(v17 + 8) = a1;
      if ( !a2 )
        goto LABEL_54;
    }
    v22 = a4;
LABEL_33:
    while ( 2 )
    {
      v22 = (_QWORD *)*v22;
      if ( v22 )
      {
        while ( *((_BYTE *)v22 + 40) )
        {
          if ( !v8 )
            goto LABEL_33;
          if ( v22[1] == *(_QWORD *)(v17 + 8) )
            goto LABEL_37;
          v22 = (_QWORD *)*v22;
          if ( !v22 )
            goto LABEL_42;
        }
        if ( v8 || v22[1] != *(_QWORD *)(v17 + 8) )
          continue;
LABEL_37:
        if ( v22[2] != *(_QWORD *)(v17 + 16) )
          continue;
        v19 = (__m128i *)v22[3];
        v21 = v22[4];
      }
      else
      {
LABEL_42:
        v31 = v17;
        v19 = sub_7E32B0(a1, v8, a2);
        v21 = sub_7E3470(a1, v8);
        v17 = v31;
      }
      break;
    }
  }
  else
  {
    *(_WORD *)(a3 + 136) = *a6;
    if ( v8 )
    {
      *(_QWORD *)(v17 + 8) = v8;
      goto LABEL_27;
    }
    *(_BYTE *)(v17 + 40) = 0;
    *(_QWORD *)(v17 + 8) = a1;
LABEL_54:
    v32 = v17;
    v19 = *(__m128i **)(v15 + 192);
    v21 = sub_7E3470(a1, 0);
    v17 = v32;
  }
LABEL_43:
  *(_QWORD *)(v17 + 32) = v21;
  v23 = a4;
  *(_QWORD *)(v17 + 24) = v19;
  if ( *a4 )
    v23 = (__int64 *)*a5;
  *v23 = v17;
  *a5 = v17;
  return a5;
}
