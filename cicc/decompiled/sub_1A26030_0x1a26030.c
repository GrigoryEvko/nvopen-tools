// Function: sub_1A26030
// Address: 0x1a26030
//
_QWORD *__fastcall sub_1A26030(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r15
  __int64 v5; // rbx
  __int64 v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v8; // rcx
  __int64 v9; // r13
  _QWORD *v10; // rax
  _QWORD *v11; // r12
  _QWORD *v12; // rdx
  _QWORD *v13; // r13
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rdx
  _QWORD *v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rcx

  v3 = a1;
  v5 = (a2 - (__int64)a1) >> 5;
  v6 = (a2 - (__int64)a1) >> 3;
  if ( v5 > 0 )
  {
    v7 = &a1[4 * v5];
    while ( 1 )
    {
      v8 = *(_QWORD **)(a3 + 16);
      v9 = *(_QWORD *)(*v3 - 48LL);
      v10 = *(_QWORD **)(a3 + 8);
      if ( v8 == v10 )
      {
        v11 = &v8[*(unsigned int *)(a3 + 28)];
        if ( v8 == v11 )
        {
          v13 = *(_QWORD **)(a3 + 16);
          v12 = v13;
        }
        else
        {
          v12 = *(_QWORD **)(a3 + 16);
          do
          {
            if ( v9 == *v12 )
              break;
            ++v12;
          }
          while ( v11 != v12 );
          v13 = &v8[*(unsigned int *)(a3 + 28)];
        }
      }
      else
      {
        v11 = &v8[*(unsigned int *)(a3 + 24)];
        v12 = sub_16CC9F0(a3, *(_QWORD *)(*v3 - 48LL));
        if ( v9 == *v12 )
        {
          v8 = *(_QWORD **)(a3 + 16);
          v10 = *(_QWORD **)(a3 + 8);
          if ( v8 == v10 )
            v19 = *(unsigned int *)(a3 + 28);
          else
            v19 = *(unsigned int *)(a3 + 24);
          v13 = &v8[v19];
        }
        else
        {
          v8 = *(_QWORD **)(a3 + 16);
          v10 = *(_QWORD **)(a3 + 8);
          if ( v8 != v10 )
          {
            v13 = &v8[*(unsigned int *)(a3 + 24)];
            if ( v11 != v13 )
              return v3;
            v15 = *(_QWORD *)(v3[1] - 48LL);
            goto LABEL_18;
          }
          v12 = &v8[*(unsigned int *)(a3 + 28)];
          v13 = v12;
        }
      }
      for ( ; v13 != v12; ++v12 )
      {
        if ( *v12 < 0xFFFFFFFFFFFFFFFELL )
          break;
      }
      if ( v11 != v12 )
        return v3;
      v15 = *(_QWORD *)(v3[1] - 48LL);
      if ( v8 == v10 )
      {
        v17 = *(unsigned int *)(a3 + 28);
        if ( v8 == &v8[v17] )
        {
LABEL_46:
          v10 = &v8[v17];
          v18 = &v8[v17];
        }
        else
        {
          while ( v15 != *v10 )
          {
            if ( &v8[v17] == ++v10 )
              goto LABEL_46;
          }
          v18 = &v8[v17];
        }
        goto LABEL_36;
      }
LABEL_18:
      v10 = sub_16CC9F0(a3, v15);
      if ( *v10 == v15 )
      {
        v20 = *(_QWORD *)(a3 + 16);
        if ( v20 == *(_QWORD *)(a3 + 8) )
          v21 = *(unsigned int *)(a3 + 28);
        else
          v21 = *(unsigned int *)(a3 + 24);
        v18 = (_QWORD *)(v20 + 8 * v21);
      }
      else
      {
        v16 = *(_QWORD *)(a3 + 16);
        if ( v16 != *(_QWORD *)(a3 + 8) )
        {
          v10 = (_QWORD *)(v16 + 8LL * *(unsigned int *)(a3 + 24));
          goto LABEL_21;
        }
        v10 = (_QWORD *)(v16 + 8LL * *(unsigned int *)(a3 + 28));
        v18 = v10;
      }
LABEL_36:
      while ( v18 != v10 )
      {
        if ( *v10 < 0xFFFFFFFFFFFFFFFELL )
          break;
        ++v10;
      }
LABEL_21:
      if ( v10 != v13 )
        return ++v3;
      if ( sub_1A25850(a3, *(_QWORD *)(v3[2] - 48LL)) )
      {
        v3 += 2;
        return v3;
      }
      if ( sub_1A25850(a3, *(_QWORD *)(v3[3] - 48LL)) )
      {
        v3 += 3;
        return v3;
      }
      v3 += 4;
      if ( v7 == v3 )
      {
        v6 = (a2 - (__int64)v3) >> 3;
        break;
      }
    }
  }
  if ( v6 == 2 )
  {
LABEL_53:
    if ( sub_1A25850(a3, *(_QWORD *)(*v3 - 48LL)) )
      return v3;
    ++v3;
    goto LABEL_55;
  }
  if ( v6 == 3 )
  {
    if ( sub_1A25850(a3, *(_QWORD *)(*v3 - 48LL)) )
      return v3;
    ++v3;
    goto LABEL_53;
  }
  if ( v6 != 1 )
    return (_QWORD *)a2;
LABEL_55:
  if ( !sub_1A25850(a3, *(_QWORD *)(*v3 - 48LL)) )
    return (_QWORD *)a2;
  return v3;
}
