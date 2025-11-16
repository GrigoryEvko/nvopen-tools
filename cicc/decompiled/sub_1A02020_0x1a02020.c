// Function: sub_1A02020
// Address: 0x1a02020
//
_QWORD *__fastcall sub_1A02020(_QWORD *a1, _QWORD *a2, __int64 *a3)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // r12
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // r14
  _QWORD *v11; // r8
  __int64 v12; // rsi
  __int64 v13; // r10
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rcx
  __int64 v18; // rcx

  v5 = a2[9];
  v6 = a2[6];
  v7 = a2[7];
  v8 = (_QWORD *)a2[2];
  v9 = (_QWORD *)a2[4];
  v10 = a2[8];
  v11 = (_QWORD *)a2[3];
  v12 = a2[5];
  v13 = (v6 - v7) >> 3;
  v14 = ((((v5 - v12) >> 3) - 1) << 6) + v13 + v9 - v8;
  v15 = v14 >> 2;
  if ( v14 >> 2 > 0 )
  {
    v16 = *a3;
    while ( *v8 != v16 )
    {
      if ( v9 == ++v8 )
      {
        v8 = *(_QWORD **)(v12 + 8);
        v12 += 8;
        v9 = v8 + 64;
        v11 = v8;
        if ( v16 == *v8 )
          goto LABEL_15;
      }
      else if ( v16 == *v8 )
      {
        goto LABEL_15;
      }
      if ( v9 == ++v8 )
      {
        v8 = *(_QWORD **)(v12 + 8);
        v12 += 8;
        v9 = v8 + 64;
        v11 = v8;
      }
      if ( v16 == *v8 )
        break;
      if ( v9 == ++v8 )
      {
        v8 = *(_QWORD **)(v12 + 8);
        v12 += 8;
        v9 = v8 + 64;
        v11 = v8;
      }
      if ( v16 == *v8 )
        break;
      if ( v9 == ++v8 )
      {
        v8 = *(_QWORD **)(v12 + 8);
        v12 += 8;
        v9 = v8 + 64;
        v11 = v8;
        if ( !--v15 )
        {
LABEL_17:
          v14 = ((((v5 - v12) >> 3) - 1) << 6) + v13 + v9 - v8;
          goto LABEL_18;
        }
      }
      else if ( !--v15 )
      {
        goto LABEL_17;
      }
    }
    goto LABEL_15;
  }
LABEL_18:
  switch ( v14 )
  {
    case 2LL:
      v18 = *a3;
LABEL_29:
      if ( v18 == *v8 )
        goto LABEL_15;
      if ( v9 == ++v8 )
      {
        v8 = *(_QWORD **)(v12 + 8);
        v12 += 8;
        v9 = v8 + 64;
        v11 = v8;
      }
      goto LABEL_26;
    case 3LL:
      v18 = *a3;
      if ( *v8 == *a3 )
        goto LABEL_15;
      if ( v9 == ++v8 )
      {
        v8 = *(_QWORD **)(v12 + 8);
        v12 += 8;
        v9 = v8 + 64;
        v11 = v8;
      }
      goto LABEL_29;
    case 1LL:
      v18 = *a3;
LABEL_26:
      if ( v18 != *v8 )
        break;
LABEL_15:
      *a1 = v8;
      a1[1] = v11;
      a1[2] = v9;
      a1[3] = v12;
      return a1;
  }
  *a1 = v6;
  a1[1] = v7;
  a1[2] = v10;
  a1[3] = v5;
  return a1;
}
