// Function: sub_D58930
// Address: 0xd58930
//
_QWORD *__fastcall sub_D58930(_QWORD *a1, __int64 a2, _QWORD *a3, __int64 *a4)
{
  __int64 v8; // r13
  __int64 v9; // rcx
  _QWORD *v10; // r14
  _QWORD *v11; // r15
  _QWORD *v12; // r9
  _QWORD *v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rsi
  __int64 v17; // r8
  __int64 v18; // rsi
  _QWORD *v19; // rsi
  _QWORD **v20; // r8
  __int64 v21; // r8
  __int64 v23; // rsi
  _QWORD *v24; // [rsp+0h] [rbp-30h]

  v8 = a3[3];
  v9 = *(_QWORD *)(a2 + 24);
  v10 = (_QWORD *)*a3;
  v11 = (_QWORD *)a3[1];
  v12 = *(_QWORD **)(a2 + 8);
  v24 = (_QWORD *)a3[2];
  v13 = *(_QWORD **)(a2 + 16);
  v14 = *(_QWORD **)a2;
  v15 = v10 - v11;
  v16 = (((__int64)v13 - *(_QWORD *)a2) >> 3) + v15 + ((((v8 - v9) >> 3) - 1) << 6);
  v17 = v16 >> 2;
  if ( v16 >> 2 <= 0 )
  {
LABEL_27:
    switch ( v16 )
    {
      case 2LL:
        v23 = *a4;
        break;
      case 3LL:
        v23 = *a4;
        if ( *v14 == *a4 )
          goto LABEL_15;
        if ( v13 == ++v14 )
        {
          v14 = *(_QWORD **)(v9 + 8);
          v9 += 8;
          v13 = v14 + 64;
          v12 = v14;
        }
        break;
      case 1LL:
        v23 = *a4;
LABEL_33:
        if ( *v14 == v23 )
          goto LABEL_15;
LABEL_30:
        v13 = v24;
        v9 = v8;
        v12 = v11;
        v14 = v10;
        goto LABEL_15;
      default:
        goto LABEL_30;
    }
    if ( v23 == *v14 )
      goto LABEL_15;
    if ( v13 == ++v14 )
    {
      v14 = *(_QWORD **)(v9 + 8);
      v9 += 8;
      v13 = v14 + 64;
      v12 = v14;
    }
    goto LABEL_33;
  }
  v18 = *a4;
  while ( *v14 != v18 )
  {
    if ( ++v14 == v13 )
    {
      v14 = *(_QWORD **)(v9 + 8);
      v9 += 8;
      v13 = v14 + 64;
      v12 = v14;
      if ( v18 == *v14 )
        break;
    }
    else if ( v18 == *v14 )
    {
      break;
    }
    if ( ++v14 == v13 )
    {
      v14 = *(_QWORD **)(v9 + 8);
      v9 += 8;
      v13 = v14 + 64;
      v12 = v14;
    }
    if ( v18 == *v14 )
      break;
    if ( ++v14 == v13 )
    {
      v14 = *(_QWORD **)(v9 + 8);
      v9 += 8;
      v13 = v14 + 64;
      v12 = v14;
    }
    if ( v18 == *v14 )
      break;
    if ( v13 == ++v14 )
    {
      v14 = *(_QWORD **)(v9 + 8);
      v9 += 8;
      v13 = v14 + 64;
      v12 = v14;
      if ( !--v17 )
      {
LABEL_26:
        v16 = ((((v8 - v9) >> 3) - 1) << 6) + v15 + v13 - v14;
        goto LABEL_27;
      }
    }
    else if ( !--v17 )
    {
      goto LABEL_26;
    }
  }
LABEL_15:
  *(_QWORD *)a2 = v14;
  *(_QWORD *)(a2 + 8) = v12;
  *(_QWORD *)(a2 + 16) = v13;
  *(_QWORD *)(a2 + 24) = v9;
  if ( (_QWORD *)*a3 != v14 )
  {
    v19 = v14 + 1;
    *(_QWORD *)a2 = v14 + 1;
    if ( v13 == v14 + 1 )
    {
      *(_QWORD *)(a2 + 24) = v9 + 8;
      v19 = *(_QWORD **)(v9 + 8);
      *(_QWORD *)(a2 + 8) = v19;
      *(_QWORD *)(a2 + 16) = v19 + 64;
      *(_QWORD *)a2 = v19;
    }
    while ( (_QWORD *)*a3 != v19 )
    {
      while ( 1 )
      {
        if ( *v19 != *a4 )
        {
          *v14++ = *v19;
          if ( v14 == v13 )
          {
            v14 = *(_QWORD **)(v9 + 8);
            v19 = *(_QWORD **)a2;
            v9 += 8;
            v13 = v14 + 64;
            v12 = v14;
          }
          else
          {
            v19 = *(_QWORD **)a2;
          }
        }
        *(_QWORD *)a2 = ++v19;
        if ( v19 != *(_QWORD **)(a2 + 16) )
          break;
        v20 = (_QWORD **)(*(_QWORD *)(a2 + 24) + 8LL);
        *(_QWORD *)(a2 + 24) = v20;
        v19 = *v20;
        v21 = (__int64)(*v20 + 64);
        *(_QWORD *)(a2 + 8) = v19;
        *(_QWORD *)(a2 + 16) = v21;
        *(_QWORD *)a2 = v19;
        if ( (_QWORD *)*a3 == v19 )
          goto LABEL_24;
      }
    }
  }
LABEL_24:
  *a1 = v14;
  a1[1] = v12;
  a1[2] = v13;
  a1[3] = v9;
  return a1;
}
