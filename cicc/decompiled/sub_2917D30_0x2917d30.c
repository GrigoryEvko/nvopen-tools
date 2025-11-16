// Function: sub_2917D30
// Address: 0x2917d30
//
__int64 *__fastcall sub_2917D30(__int64 a1, __int64 a2)
{
  __int64 *v4; // r14
  __int64 v5; // rbx
  __int64 *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 *v9; // rbx
  _QWORD *v10; // rsi
  _QWORD *v11; // rax
  _QWORD *v12; // rcx
  _QWORD *v13; // rdx
  __int64 *i; // rbx
  __int64 v15; // rsi
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  __int64 *result; // rax
  __int64 v19; // rbx
  __int64 v20; // r8
  __int64 *v21; // r9
  _QWORD *v22; // rdx
  __int64 v23; // r8
  _QWORD *v24; // rdx
  __int64 v25; // r8
  __int64 *v26; // rdx
  __int64 *v27; // rax
  __int64 *v28; // rax
  __int64 *v29; // rax
  char v30; // al
  __int64 v31; // r8
  _QWORD *v32; // rsi
  __int64 v33; // rax
  _QWORD *v34; // rax
  char v35; // di
  _QWORD *v36; // rcx
  _QWORD *v37; // rdx
  _QWORD *v38; // rcx
  _QWORD *v39; // rdx
  __int64 v40; // r8

  v4 = *(__int64 **)a1;
  v5 = 8LL * *(unsigned int *)(a1 + 8);
  v6 = (__int64 *)(*(_QWORD *)a1 + v5);
  v7 = v5 >> 3;
  v8 = v5 >> 5;
  if ( !v8 )
  {
LABEL_47:
    switch ( v7 )
    {
      case 2LL:
        v35 = *(_BYTE *)(a2 + 28);
        break;
      case 3LL:
        v35 = *(_BYTE *)(a2 + 28);
        if ( v35 )
        {
          v32 = *(_QWORD **)(a2 + 8);
          v33 = *(unsigned int *)(a2 + 20);
          v36 = &v32[v33];
          v37 = v32;
          if ( v32 != v36 )
          {
            while ( *v4 != *v37 )
            {
              if ( v36 == ++v37 )
                goto LABEL_81;
            }
            goto LABEL_8;
          }
          v40 = v4[1];
          ++v4;
          goto LABEL_74;
        }
        if ( sub_C8CA60(a2, *v4) )
          goto LABEL_8;
        v35 = *(_BYTE *)(a2 + 28);
LABEL_81:
        ++v4;
        break;
      case 1LL:
        v30 = *(_BYTE *)(a2 + 28);
        goto LABEL_58;
      default:
LABEL_50:
        v4 = v6;
        goto LABEL_16;
    }
    v40 = *v4;
    if ( !v35 )
    {
      if ( sub_C8CA60(a2, *v4) )
        goto LABEL_8;
      v30 = *(_BYTE *)(a2 + 28);
LABEL_86:
      ++v4;
LABEL_58:
      v31 = *v4;
      if ( !v30 )
      {
        if ( sub_C8CA60(a2, *v4) )
          goto LABEL_8;
        goto LABEL_50;
      }
      v32 = *(_QWORD **)(a2 + 8);
      v33 = *(unsigned int *)(a2 + 20);
LABEL_60:
      v34 = &v32[v33];
      if ( v32 != v34 )
      {
        while ( v31 != *v32 )
        {
          if ( v34 == ++v32 )
            goto LABEL_50;
        }
        goto LABEL_8;
      }
      goto LABEL_50;
    }
    v32 = *(_QWORD **)(a2 + 8);
    v33 = *(unsigned int *)(a2 + 20);
LABEL_74:
    v38 = &v32[v33];
    v39 = v32;
    if ( v32 == v38 )
    {
      v31 = v4[1];
      ++v4;
      goto LABEL_60;
    }
    do
    {
      if ( *v39 == v40 )
        goto LABEL_8;
      ++v39;
    }
    while ( v38 != v39 );
    v30 = 1;
    goto LABEL_86;
  }
  v9 = &v4[4 * v8];
  while ( 1 )
  {
    if ( *(_BYTE *)(a2 + 28) )
    {
      v10 = *(_QWORD **)(a2 + 8);
      v11 = &v10[*(unsigned int *)(a2 + 20)];
      v12 = v10;
      if ( v10 != v11 )
      {
        v13 = *(_QWORD **)(a2 + 8);
        while ( *v4 != *v13 )
        {
          if ( v11 == ++v13 )
            goto LABEL_23;
        }
        goto LABEL_8;
      }
LABEL_23:
      v20 = v4[1];
      v21 = v4 + 1;
      goto LABEL_24;
    }
    if ( sub_C8CA60(a2, *v4) )
      goto LABEL_8;
    v20 = v4[1];
    v21 = v4 + 1;
    if ( *(_BYTE *)(a2 + 28) )
    {
      v10 = *(_QWORD **)(a2 + 8);
      v12 = v10;
      v11 = &v10[*(unsigned int *)(a2 + 20)];
LABEL_24:
      if ( v11 != v10 )
      {
        v22 = v10;
        while ( v20 != *v22 )
        {
          if ( v11 == ++v22 )
            goto LABEL_29;
        }
        goto LABEL_28;
      }
LABEL_29:
      v23 = v4[2];
      v21 = v4 + 2;
      goto LABEL_30;
    }
    v28 = sub_C8CA60(a2, v4[1]);
    v21 = v4 + 1;
    if ( v28 )
      goto LABEL_28;
    v23 = v4[2];
    v21 = v4 + 2;
    if ( *(_BYTE *)(a2 + 28) )
    {
      v10 = *(_QWORD **)(a2 + 8);
      v12 = v10;
      v11 = &v10[*(unsigned int *)(a2 + 20)];
LABEL_30:
      if ( v10 != v11 )
      {
        v24 = v10;
        while ( v23 != *v24 )
        {
          if ( ++v24 == v11 )
            goto LABEL_38;
        }
LABEL_28:
        v4 = v21;
        goto LABEL_8;
      }
LABEL_38:
      v25 = v4[3];
      v26 = v4 + 3;
      goto LABEL_39;
    }
    v29 = sub_C8CA60(a2, v4[2]);
    v21 = v4 + 2;
    if ( v29 )
      goto LABEL_28;
    v25 = v4[3];
    v26 = v4 + 3;
    if ( !*(_BYTE *)(a2 + 28) )
    {
      v27 = sub_C8CA60(a2, v4[3]);
      v26 = v4 + 3;
      if ( v27 )
        goto LABEL_43;
      goto LABEL_45;
    }
    v10 = *(_QWORD **)(a2 + 8);
    v12 = v10;
    v11 = &v10[*(unsigned int *)(a2 + 20)];
LABEL_39:
    if ( v10 != v11 )
      break;
LABEL_45:
    v4 += 4;
    if ( v9 == v4 )
    {
      v7 = v6 - v4;
      goto LABEL_47;
    }
  }
  while ( *v12 != v25 )
  {
    if ( ++v12 == v11 )
      goto LABEL_45;
  }
LABEL_43:
  v4 = v26;
LABEL_8:
  if ( v6 != v4 )
  {
    for ( i = v4 + 1; v6 != i; ++v4 )
    {
LABEL_10:
      v15 = *i;
      if ( *(_BYTE *)(a2 + 28) )
      {
        v16 = *(_QWORD **)(a2 + 8);
        v17 = &v16[*(unsigned int *)(a2 + 20)];
        if ( v16 != v17 )
        {
          while ( v15 != *v16 )
          {
            if ( v17 == ++v16 )
              goto LABEL_21;
          }
LABEL_15:
          if ( v6 == ++i )
            break;
          goto LABEL_10;
        }
      }
      else
      {
        if ( sub_C8CA60(a2, v15) )
          goto LABEL_15;
        v15 = *i;
      }
LABEL_21:
      ++i;
      *v4 = v15;
    }
  }
LABEL_16:
  result = *(__int64 **)a1;
  v19 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v6;
  if ( v6 != (__int64 *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)) )
  {
    memmove(v4, v6, *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v6);
    result = *(__int64 **)a1;
  }
  *(_DWORD *)(a1 + 8) = ((char *)v4 + v19 - (char *)result) >> 3;
  return result;
}
