// Function: sub_1A25C30
// Address: 0x1a25c30
//
__int64 *__fastcall sub_1A25C30(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // r13
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 *v7; // rbx
  _QWORD *v8; // rax
  _QWORD *v9; // rsi
  __int64 v10; // r12
  _QWORD *v11; // rcx
  _QWORD *v12; // r15
  _QWORD *v13; // rdx
  _QWORD *v14; // r12
  __int64 v16; // r15
  _QWORD *v17; // r15
  __int64 v18; // rdi
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rcx

  v4 = a1;
  v5 = (a2 - (__int64)a1) >> 5;
  v6 = (a2 - (__int64)a1) >> 3;
  if ( v5 > 0 )
  {
    v7 = &a1[4 * v5];
    while ( 1 )
    {
      v8 = *(_QWORD **)(a3 + 8);
      v9 = *(_QWORD **)(a3 + 16);
      v10 = *v4;
      v11 = v8;
      if ( v9 == v8 )
      {
        v12 = &v9[*(unsigned int *)(a3 + 28)];
        if ( v9 == v12 )
        {
          v14 = *(_QWORD **)(a3 + 16);
          v13 = v14;
        }
        else
        {
          v13 = *(_QWORD **)(a3 + 16);
          do
          {
            if ( v10 == *v13 )
              break;
            ++v13;
          }
          while ( v12 != v13 );
          v14 = &v9[*(unsigned int *)(a3 + 28)];
        }
      }
      else
      {
        v12 = &v9[*(unsigned int *)(a3 + 24)];
        v13 = sub_16CC9F0(a3, *v4);
        if ( v10 == *v13 )
        {
          v8 = *(_QWORD **)(a3 + 8);
          v9 = *(_QWORD **)(a3 + 16);
          v11 = v8;
          if ( v9 == v8 )
            v23 = *(unsigned int *)(a3 + 28);
          else
            v23 = *(unsigned int *)(a3 + 24);
          v14 = &v9[v23];
        }
        else
        {
          v8 = *(_QWORD **)(a3 + 8);
          v9 = *(_QWORD **)(a3 + 16);
          v11 = v8;
          if ( v9 != v8 )
          {
            v14 = &v9[*(unsigned int *)(a3 + 24)];
            if ( v12 != v14 )
              return v4;
            v16 = v4[1];
            goto LABEL_18;
          }
          v13 = &v9[*(unsigned int *)(a3 + 28)];
          v14 = v13;
        }
      }
      while ( v14 != v13 && *v13 >= 0xFFFFFFFFFFFFFFFELL )
        ++v13;
      if ( v13 != v12 )
        return v4;
      v16 = v4[1];
      if ( v9 == v8 )
      {
        v18 = *(unsigned int *)(a3 + 28);
        if ( &v9[v18] == v9 )
        {
LABEL_62:
          v11 = &v9[v18];
          v17 = &v9[v18];
        }
        else
        {
          while ( v16 != *v11 )
          {
            if ( &v9[v18] == ++v11 )
              goto LABEL_62;
          }
          v17 = &v9[v18];
        }
        goto LABEL_29;
      }
LABEL_18:
      v11 = sub_16CC9F0(a3, v16);
      if ( *v11 == v16 )
      {
        v9 = *(_QWORD **)(a3 + 16);
        v8 = *(_QWORD **)(a3 + 8);
        if ( v9 == v8 )
          v24 = *(unsigned int *)(a3 + 28);
        else
          v24 = *(unsigned int *)(a3 + 24);
        v17 = &v9[v24];
      }
      else
      {
        v9 = *(_QWORD **)(a3 + 16);
        v8 = *(_QWORD **)(a3 + 8);
        if ( v9 != v8 )
        {
          v17 = &v9[*(unsigned int *)(a3 + 24)];
          if ( v14 != v17 )
            return ++v4;
          v19 = v4[2];
          goto LABEL_32;
        }
        v11 = &v9[*(unsigned int *)(a3 + 28)];
        v17 = v11;
      }
LABEL_29:
      while ( v17 != v11 && *v11 >= 0xFFFFFFFFFFFFFFFELL )
        ++v11;
      if ( v14 != v11 )
        return ++v4;
      v19 = v4[2];
      if ( v9 == v8 )
      {
        v21 = *(unsigned int *)(a3 + 28);
        if ( v9 == &v9[v21] )
        {
LABEL_63:
          v8 = &v9[v21];
          v22 = &v9[v21];
        }
        else
        {
          while ( v19 != *v8 )
          {
            if ( &v9[v21] == ++v8 )
              goto LABEL_63;
          }
          v22 = &v9[v21];
        }
        goto LABEL_51;
      }
LABEL_32:
      v8 = sub_16CC9F0(a3, v19);
      if ( *v8 == v19 )
      {
        v25 = *(_QWORD *)(a3 + 16);
        if ( v25 == *(_QWORD *)(a3 + 8) )
          v26 = *(unsigned int *)(a3 + 28);
        else
          v26 = *(unsigned int *)(a3 + 24);
        v22 = (_QWORD *)(v25 + 8 * v26);
      }
      else
      {
        v20 = *(_QWORD *)(a3 + 16);
        if ( v20 != *(_QWORD *)(a3 + 8) )
        {
          v8 = (_QWORD *)(v20 + 8LL * *(unsigned int *)(a3 + 24));
          goto LABEL_35;
        }
        v8 = (_QWORD *)(v20 + 8LL * *(unsigned int *)(a3 + 28));
        v22 = v8;
      }
LABEL_51:
      while ( v22 != v8 && *v8 >= 0xFFFFFFFFFFFFFFFELL )
        ++v8;
LABEL_35:
      if ( v17 != v8 )
      {
        v4 += 2;
        return v4;
      }
      if ( sub_1A25850(a3, v4[3]) )
      {
        v4 += 3;
        return v4;
      }
      v4 += 4;
      if ( v7 == v4 )
      {
        v6 = (a2 - (__int64)v4) >> 3;
        break;
      }
    }
  }
  if ( v6 == 2 )
  {
LABEL_72:
    if ( sub_1A25850(a3, *v4) )
      return v4;
    ++v4;
    goto LABEL_74;
  }
  if ( v6 == 3 )
  {
    if ( sub_1A25850(a3, *v4) )
      return v4;
    ++v4;
    goto LABEL_72;
  }
  if ( v6 != 1 )
    return (__int64 *)a2;
LABEL_74:
  if ( !sub_1A25850(a3, *v4) )
    return (__int64 *)a2;
  return v4;
}
