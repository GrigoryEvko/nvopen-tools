// Function: sub_2917420
// Address: 0x2917420
//
char *__fastcall sub_2917420(__int64 a1, __int64 a2)
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
  __int64 v15; // rcx
  __int64 v16; // rsi
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  char *result; // rax
  __int64 v20; // rbx
  __int64 *v21; // r9
  __int64 v22; // r8
  _QWORD *v23; // rdx
  __int64 v24; // r8
  _QWORD *v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // r8
  __int64 *v29; // rax
  __int64 *v30; // rax
  __int64 v31; // rax
  __int64 *v32; // rax
  __int64 v33; // rax
  char v34; // al
  __int64 v35; // r8
  _QWORD *v36; // rsi
  __int64 v37; // rax
  _QWORD *v38; // rax
  char v39; // di
  _QWORD *v40; // rcx
  _QWORD *v41; // rdx
  _QWORD *v42; // rcx
  _QWORD *v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // r8

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
        v39 = *(_BYTE *)(a2 + 28);
        break;
      case 3LL:
        v39 = *(_BYTE *)(a2 + 28);
        if ( v39 )
        {
          v36 = *(_QWORD **)(a2 + 8);
          v37 = *(unsigned int *)(a2 + 20);
          v40 = &v36[v37];
          v41 = v36;
          if ( v36 != v40 )
          {
            while ( *(_QWORD *)(*v4 - 64) != *v41 )
            {
              if ( v40 == ++v41 )
                goto LABEL_81;
            }
            goto LABEL_8;
          }
          v45 = v4[1];
          ++v4;
          v46 = *(_QWORD *)(v45 - 64);
          goto LABEL_74;
        }
        if ( sub_C8CA60(a2, *(_QWORD *)(*v4 - 64)) )
          goto LABEL_8;
        v39 = *(_BYTE *)(a2 + 28);
LABEL_81:
        ++v4;
        break;
      case 1LL:
        v34 = *(_BYTE *)(a2 + 28);
        goto LABEL_58;
      default:
LABEL_50:
        v4 = v6;
        goto LABEL_16;
    }
    v46 = *(_QWORD *)(*v4 - 64);
    if ( !v39 )
    {
      if ( sub_C8CA60(a2, *(_QWORD *)(*v4 - 64)) )
        goto LABEL_8;
      v34 = *(_BYTE *)(a2 + 28);
LABEL_86:
      ++v4;
LABEL_58:
      v35 = *(_QWORD *)(*v4 - 64);
      if ( !v34 )
      {
        if ( sub_C8CA60(a2, *(_QWORD *)(*v4 - 64)) )
          goto LABEL_8;
        goto LABEL_50;
      }
      v36 = *(_QWORD **)(a2 + 8);
      v37 = *(unsigned int *)(a2 + 20);
LABEL_60:
      v38 = &v36[v37];
      if ( v36 != v38 )
      {
        while ( *v36 != v35 )
        {
          if ( v38 == ++v36 )
            goto LABEL_50;
        }
        goto LABEL_8;
      }
      goto LABEL_50;
    }
    v36 = *(_QWORD **)(a2 + 8);
    v37 = *(unsigned int *)(a2 + 20);
LABEL_74:
    v42 = &v36[v37];
    v43 = v36;
    if ( v36 == v42 )
    {
      v44 = v4[1];
      ++v4;
      v35 = *(_QWORD *)(v44 - 64);
      goto LABEL_60;
    }
    do
    {
      if ( *v43 == v46 )
        goto LABEL_8;
      ++v43;
    }
    while ( v42 != v43 );
    v34 = 1;
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
        while ( *(_QWORD *)(*v4 - 64) != *v13 )
        {
          if ( v11 == ++v13 )
            goto LABEL_23;
        }
        goto LABEL_8;
      }
LABEL_23:
      v21 = v4 + 1;
      v22 = *(_QWORD *)(v4[1] - 64);
      goto LABEL_24;
    }
    if ( sub_C8CA60(a2, *(_QWORD *)(*v4 - 64)) )
      goto LABEL_8;
    v26 = v4[1];
    v21 = v4 + 1;
    v22 = *(_QWORD *)(v26 - 64);
    if ( *(_BYTE *)(a2 + 28) )
    {
      v10 = *(_QWORD **)(a2 + 8);
      v12 = v10;
      v11 = &v10[*(unsigned int *)(a2 + 20)];
LABEL_24:
      if ( v11 != v10 )
      {
        v23 = v10;
        while ( *v23 != v22 )
        {
          if ( ++v23 == v11 )
            goto LABEL_29;
        }
        goto LABEL_28;
      }
LABEL_29:
      v21 = v4 + 2;
      v24 = *(_QWORD *)(v4[2] - 64);
      goto LABEL_30;
    }
    v30 = sub_C8CA60(a2, *(_QWORD *)(v26 - 64));
    v21 = v4 + 1;
    if ( v30 )
      goto LABEL_28;
    v31 = v4[2];
    v21 = v4 + 2;
    v24 = *(_QWORD *)(v31 - 64);
    if ( *(_BYTE *)(a2 + 28) )
    {
      v10 = *(_QWORD **)(a2 + 8);
      v12 = v10;
      v11 = &v10[*(unsigned int *)(a2 + 20)];
LABEL_30:
      if ( v11 != v10 )
      {
        v25 = v10;
        while ( v24 != *v25 )
        {
          if ( v11 == ++v25 )
            goto LABEL_38;
        }
LABEL_28:
        v4 = v21;
        goto LABEL_8;
      }
LABEL_38:
      v27 = v4 + 3;
      v28 = *(_QWORD *)(v4[3] - 64);
      goto LABEL_39;
    }
    v32 = sub_C8CA60(a2, *(_QWORD *)(v31 - 64));
    v21 = v4 + 2;
    if ( v32 )
      goto LABEL_28;
    v33 = v4[3];
    v27 = v4 + 3;
    v28 = *(_QWORD *)(v33 - 64);
    if ( !*(_BYTE *)(a2 + 28) )
    {
      v29 = sub_C8CA60(a2, *(_QWORD *)(v33 - 64));
      v27 = v4 + 3;
      if ( v29 )
        goto LABEL_43;
      goto LABEL_45;
    }
    v10 = *(_QWORD **)(a2 + 8);
    v12 = v10;
    v11 = &v10[*(unsigned int *)(a2 + 20)];
LABEL_39:
    if ( v11 != v10 )
      break;
LABEL_45:
    v4 += 4;
    if ( v9 == v4 )
    {
      v7 = v6 - v4;
      goto LABEL_47;
    }
  }
  while ( *v12 != v28 )
  {
    if ( v11 == ++v12 )
      goto LABEL_45;
  }
LABEL_43:
  v4 = v27;
LABEL_8:
  if ( v6 != v4 )
  {
    for ( i = v4 + 1; v6 != i; ++v4 )
    {
LABEL_10:
      v15 = *i;
      v16 = *(_QWORD *)(*i - 64);
      if ( *(_BYTE *)(a2 + 28) )
      {
        v17 = *(_QWORD **)(a2 + 8);
        v18 = &v17[*(unsigned int *)(a2 + 20)];
        if ( v17 != v18 )
        {
          while ( v16 != *v17 )
          {
            if ( v18 == ++v17 )
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
        if ( sub_C8CA60(a2, v16) )
          goto LABEL_15;
        v15 = *i;
      }
LABEL_21:
      ++i;
      *v4 = v15;
    }
  }
LABEL_16:
  result = *(char **)a1;
  v20 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v6;
  if ( v6 != (__int64 *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)) )
  {
    memmove(v4, v6, *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v6);
    result = *(char **)a1;
  }
  *(_DWORD *)(a1 + 8) = ((char *)v4 + v20 - result) >> 3;
  return result;
}
