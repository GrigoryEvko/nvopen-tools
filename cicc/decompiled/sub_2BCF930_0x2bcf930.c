// Function: sub_2BCF930
// Address: 0x2bcf930
//
__int64 __fastcall sub_2BCF930(__int64 *a1, __int64 *a2, unsigned __int64 a3, char a4, __m128i a5)
{
  __int64 *v7; // r11
  __int64 v8; // r10
  __int64 v9; // rax
  __int64 *v10; // rcx
  __int64 *v11; // r10
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 i; // rax
  __int64 v27; // rdx
  __int64 j; // rax
  __int64 v29; // rdx

  v7 = &a2[a3];
  v8 = (__int64)(8 * a3) >> 5;
  v9 = (__int64)(8 * a3) >> 3;
  if ( v8 <= 0 )
  {
    v10 = a2;
LABEL_29:
    if ( v9 != 2 )
    {
      if ( v9 != 3 )
      {
        if ( v9 != 1 )
          return sub_2BCE070(*a1, a2, a3, a1[1], a4, a5);
        goto LABEL_32;
      }
      for ( i = *(_QWORD *)(*v10 + 16); i; i = *(_QWORD *)(i + 8) )
      {
        v27 = *(_QWORD *)(i + 24);
        if ( *(_BYTE *)v27 == 86 && *(_QWORD *)(v27 + 40) != *(_QWORD *)(*v10 + 40) )
          goto LABEL_37;
      }
      ++v10;
    }
    for ( j = *(_QWORD *)(*v10 + 16); j; j = *(_QWORD *)(j + 8) )
    {
      v29 = *(_QWORD *)(j + 24);
      if ( *(_BYTE *)v29 == 86 && *(_QWORD *)(v29 + 40) != *(_QWORD *)(*v10 + 40) )
        goto LABEL_37;
    }
    ++v10;
LABEL_32:
    v23 = *(_QWORD *)(*v10 + 16);
    if ( !v23 )
      return sub_2BCE070(*a1, a2, a3, a1[1], a4, a5);
    while ( 1 )
    {
      v24 = *(_QWORD *)(v23 + 24);
      if ( *(_BYTE *)v24 == 86 && *(_QWORD *)(v24 + 40) != *(_QWORD *)(*v10 + 40) )
        break;
      v23 = *(_QWORD *)(v23 + 8);
      if ( !v23 )
        return sub_2BCE070(*a1, a2, a3, a1[1], a4, a5);
    }
    goto LABEL_37;
  }
  v10 = a2;
  v11 = &a2[4 * v8];
  while ( 1 )
  {
    v12 = *(_QWORD *)(*v10 + 16);
    if ( v12 )
      break;
LABEL_9:
    v14 = v10[1];
    v15 = *(_QWORD *)(v14 + 16);
    if ( v15 )
    {
      while ( 1 )
      {
        v16 = *(_QWORD *)(v15 + 24);
        if ( *(_BYTE *)v16 == 86 && *(_QWORD *)(v16 + 40) != *(_QWORD *)(v14 + 40) )
          break;
        v15 = *(_QWORD *)(v15 + 8);
        if ( !v15 )
          goto LABEL_15;
      }
      ++v10;
      goto LABEL_37;
    }
LABEL_15:
    v17 = v10[2];
    v18 = *(_QWORD *)(v17 + 16);
    if ( v18 )
    {
      while ( 1 )
      {
        v19 = *(_QWORD *)(v18 + 24);
        if ( *(_BYTE *)v19 == 86 && *(_QWORD *)(v19 + 40) != *(_QWORD *)(v17 + 40) )
          break;
        v18 = *(_QWORD *)(v18 + 8);
        if ( !v18 )
          goto LABEL_21;
      }
      v10 += 2;
      goto LABEL_37;
    }
LABEL_21:
    v20 = v10[3];
    v21 = *(_QWORD *)(v20 + 16);
    if ( v21 )
    {
      while ( 1 )
      {
        v22 = *(_QWORD *)(v21 + 24);
        if ( *(_BYTE *)v22 == 86 && *(_QWORD *)(v22 + 40) != *(_QWORD *)(v20 + 40) )
          break;
        v21 = *(_QWORD *)(v21 + 8);
        if ( !v21 )
          goto LABEL_27;
      }
      v10 += 3;
      goto LABEL_37;
    }
LABEL_27:
    v10 += 4;
    if ( v11 == v10 )
    {
      v9 = v7 - v10;
      goto LABEL_29;
    }
  }
  while ( 1 )
  {
    v13 = *(_QWORD *)(v12 + 24);
    if ( *(_BYTE *)v13 == 86 && *(_QWORD *)(v13 + 40) != *(_QWORD *)(*v10 + 40) )
      break;
    v12 = *(_QWORD *)(v12 + 8);
    if ( !v12 )
      goto LABEL_9;
  }
LABEL_37:
  if ( v7 != v10 )
    return 0;
  return sub_2BCE070(*a1, a2, a3, a1[1], a4, a5);
}
