// Function: sub_2672B10
// Address: 0x2672b10
//
__int64 *__fastcall sub_2672B10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 *v10; // r14
  __int64 v11; // rsi
  __int64 *v12; // r13
  __int64 *result; // rax
  __int64 v14; // rdx
  __int64 *v15; // r14
  __int64 v16; // rsi
  __int64 *v17; // r13
  char v18; // di
  _QWORD *v19; // rax
  char v20; // di
  _QWORD *v21; // rax
  __int64 *v22; // rax

  v8 = *(__int64 **)(a2 + 64);
  if ( *(_BYTE *)(a2 + 84) )
    v9 = *(unsigned int *)(a2 + 76);
  else
    v9 = *(unsigned int *)(a2 + 72);
  v10 = &v8[v9];
  if ( v8 != v10 )
  {
    while ( 1 )
    {
      v11 = *v8;
      v12 = v8;
      if ( (unsigned __int64)*v8 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v10 == ++v8 )
        goto LABEL_6;
    }
    if ( v10 != v8 )
    {
      v20 = *(_BYTE *)(a1 + 84);
      if ( !v20 )
        goto LABEL_39;
LABEL_29:
      v21 = *(_QWORD **)(a1 + 64);
      a4 = *(unsigned int *)(a1 + 76);
      v9 = (__int64)&v21[a4];
      if ( v21 == (_QWORD *)v9 )
      {
LABEL_40:
        if ( (unsigned int)a4 < *(_DWORD *)(a1 + 72) )
        {
          a4 = (unsigned int)(a4 + 1);
          *(_DWORD *)(a1 + 76) = a4;
          *(_QWORD *)v9 = v11;
          v20 = *(_BYTE *)(a1 + 84);
          ++*(_QWORD *)(a1 + 56);
          goto LABEL_33;
        }
        goto LABEL_39;
      }
      while ( *v21 != v11 )
      {
        if ( (_QWORD *)v9 == ++v21 )
          goto LABEL_40;
      }
LABEL_33:
      while ( 1 )
      {
        v22 = v12 + 1;
        if ( v12 + 1 == v10 )
          break;
        v11 = *v22;
        for ( ++v12; (unsigned __int64)*v22 >= 0xFFFFFFFFFFFFFFFELL; v12 = v22 )
        {
          if ( v10 == ++v22 )
            goto LABEL_6;
          v11 = *v22;
        }
        if ( v10 == v12 )
          break;
        if ( v20 )
          goto LABEL_29;
LABEL_39:
        sub_C8CC70(a1 + 56, v11, v9, a4, a5, a6);
        v20 = *(_BYTE *)(a1 + 84);
      }
    }
  }
LABEL_6:
  result = *(__int64 **)(a2 + 16);
  if ( *(_BYTE *)(a2 + 36) )
    v14 = *(unsigned int *)(a2 + 28);
  else
    v14 = *(unsigned int *)(a2 + 24);
  v15 = &result[v14];
  if ( result != v15 )
  {
    while ( 1 )
    {
      v16 = *result;
      v17 = result;
      if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v15 == ++result )
        return result;
    }
    if ( v15 != result )
    {
      v18 = *(_BYTE *)(a1 + 36);
      if ( !v18 )
        goto LABEL_24;
LABEL_14:
      v19 = *(_QWORD **)(a1 + 16);
      a4 = *(unsigned int *)(a1 + 28);
      v14 = (__int64)&v19[a4];
      if ( v19 == (_QWORD *)v14 )
      {
LABEL_25:
        if ( (unsigned int)a4 < *(_DWORD *)(a1 + 24) )
        {
          a4 = (unsigned int)(a4 + 1);
          *(_DWORD *)(a1 + 28) = a4;
          *(_QWORD *)v14 = v16;
          v18 = *(_BYTE *)(a1 + 36);
          ++*(_QWORD *)(a1 + 8);
          goto LABEL_18;
        }
        goto LABEL_24;
      }
      while ( v16 != *v19 )
      {
        if ( (_QWORD *)v14 == ++v19 )
          goto LABEL_25;
      }
LABEL_18:
      while ( 1 )
      {
        result = v17 + 1;
        if ( v17 + 1 == v15 )
          break;
        v16 = *result;
        for ( ++v17; (unsigned __int64)*result >= 0xFFFFFFFFFFFFFFFELL; v17 = result )
        {
          if ( v15 == ++result )
            return result;
          v16 = *result;
        }
        if ( v15 == v17 )
          return result;
        if ( v18 )
          goto LABEL_14;
LABEL_24:
        sub_C8CC70(a1 + 8, v16, v14, a4, a5, a6);
        v18 = *(_BYTE *)(a1 + 36);
      }
    }
  }
  return result;
}
