// Function: sub_1DD50F0
// Address: 0x1dd50f0
//
__int64 __fastcall sub_1DD50F0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned __int64 v5; // r14
  __int64 v6; // rbx
  unsigned __int64 v7; // r9
  __int64 v8; // r12
  unsigned __int16 v9; // cx
  unsigned __int16 v10; // di
  unsigned __int16 v11; // si
  int v12; // r10d
  __int64 v13; // rax
  unsigned __int16 v14; // dx
  __int64 v15; // rax
  unsigned __int16 v16; // ax
  unsigned __int16 v17; // cx
  _WORD *v18; // rsi
  unsigned __int64 v19; // r13
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  int v22; // eax
  __int64 v23; // r12
  __int64 i; // rbx
  unsigned __int64 *v25; // r14
  unsigned __int64 v26; // rcx
  __int64 v27; // rbx
  _WORD *v28; // [rsp-40h] [rbp-40h]

  result = a2 - a1;
  if ( (__int64)(a2 - a1) <= 128 )
    return result;
  v5 = a2;
  v6 = a3;
  if ( !a3 )
    goto LABEL_24;
  v7 = a2;
  v8 = a1 + 8;
  v28 = (_WORD *)(a1 + 16);
  while ( 2 )
  {
    v9 = *(_WORD *)(a1 + 8);
    v10 = *(_WORD *)a1;
    --v6;
    v11 = *(_WORD *)(v7 - 8);
    v12 = *(_DWORD *)(a1 + 4);
    v13 = a1 + 8 * ((__int64)(((__int64)(v7 - a1) >> 3) + ((v7 - a1) >> 63)) >> 1);
    v14 = *(_WORD *)v13;
    if ( v9 >= *(_WORD *)v13 )
    {
      if ( v9 < v11 )
        goto LABEL_7;
      if ( v14 < v11 )
      {
LABEL_18:
        *(_QWORD *)a1 = *(_QWORD *)(v7 - 8);
        v16 = v10;
        *(_WORD *)(v7 - 8) = v10;
        *(_DWORD *)(v7 - 4) = v12;
        v10 = *(_WORD *)(a1 + 8);
        goto LABEL_8;
      }
LABEL_23:
      *(_QWORD *)a1 = *(_QWORD *)v13;
      *(_WORD *)v13 = v10;
      *(_DWORD *)(v13 + 4) = v12;
      v10 = *(_WORD *)(a1 + 8);
      v16 = *(_WORD *)(v7 - 8);
      goto LABEL_8;
    }
    if ( v14 < v11 )
      goto LABEL_23;
    if ( v9 < v11 )
      goto LABEL_18;
LABEL_7:
    v15 = *(_QWORD *)(a1 + 8);
    *(_WORD *)(a1 + 8) = v10;
    *(_DWORD *)(a1 + 12) = v12;
    *(_QWORD *)a1 = v15;
    v16 = *(_WORD *)(v7 - 8);
LABEL_8:
    v17 = *(_WORD *)a1;
    v18 = v28;
    v19 = v8;
    v20 = v7;
    while ( 1 )
    {
      v5 = v19;
      if ( v17 > v10 )
        goto LABEL_15;
      if ( v17 >= v16 )
      {
        v20 -= 8LL;
      }
      else
      {
        v21 = v20 - 16;
        do
        {
          v20 = v21;
          v21 -= 8LL;
        }
        while ( v17 < *(_WORD *)(v21 + 8) );
      }
      if ( v19 >= v20 )
        break;
      v22 = *((_DWORD *)v18 - 1);
      *((_QWORD *)v18 - 1) = *(_QWORD *)v20;
      *(_DWORD *)(v20 + 4) = v22;
      v16 = *(_WORD *)(v20 - 8);
      *(_WORD *)v20 = v10;
      v17 = *(_WORD *)a1;
LABEL_15:
      v10 = *v18;
      v19 += 8LL;
      v18 += 4;
    }
    sub_1DD50F0(v19, v7, v6);
    result = v19 - a1;
    if ( (__int64)(v19 - a1) > 128 )
    {
      if ( v6 )
      {
        v7 = v19;
        continue;
      }
LABEL_24:
      v23 = result >> 3;
      for ( i = ((result >> 3) - 2) >> 1; ; --i )
      {
        sub_1DD4D70(a1, i, v23, *(_QWORD *)(a1 + 8 * i));
        if ( !i )
          break;
      }
      v25 = (unsigned __int64 *)(v5 - 8);
      do
      {
        v26 = *v25;
        v27 = (__int64)v25-- - a1;
        v25[1] = *(_QWORD *)a1;
        result = sub_1DD4D70(a1, 0, v27 >> 3, v26);
      }
      while ( v27 > 8 );
    }
    return result;
  }
}
