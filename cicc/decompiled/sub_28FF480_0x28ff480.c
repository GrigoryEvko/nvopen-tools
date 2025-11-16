// Function: sub_28FF480
// Address: 0x28ff480
//
__int64 *__fastcall sub_28FF480(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // r15
  __int64 v8; // rbx
  __int64 *v9; // r12
  __int64 v10; // rax
  __int64 v11; // rbx
  char v12; // al
  __int64 *v13; // rbx
  __int64 v14; // rsi
  _QWORD *v15; // rax
  __int64 *i; // rbx
  __int64 v17; // rsi
  _QWORD *v18; // rax
  __int64 *result; // rax
  __int64 v20; // rbx
  _QWORD *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rsi
  _QWORD *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rsi
  _QWORD *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rax

  v7 = *(__int64 **)a1;
  v8 = 8LL * *(unsigned int *)(a1 + 8);
  v9 = (__int64 *)(*(_QWORD *)a1 + v8);
  v10 = v8 >> 3;
  v11 = v8 >> 5;
  if ( !v11 )
  {
LABEL_43:
    if ( v10 != 2 )
    {
      if ( v10 != 3 )
      {
        if ( v10 != 1 )
        {
LABEL_46:
          v7 = v9;
          goto LABEL_16;
        }
LABEL_64:
        if ( (unsigned __int8)sub_28FF410(a2, *v7, (__int64 *)a3, a4, a5, a6) )
          goto LABEL_8;
        goto LABEL_46;
      }
      if ( (unsigned __int8)sub_28FF410(a2, *v7, (__int64 *)a3, a4, a5, a6) )
        goto LABEL_8;
      ++v7;
    }
    if ( (unsigned __int8)sub_28FF410(a2, *v7, (__int64 *)a3, a4, a5, a6) )
      goto LABEL_8;
    ++v7;
    goto LABEL_64;
  }
  v12 = *(_BYTE *)(a2 + 28);
  v13 = &v7[4 * v11];
  while ( 1 )
  {
    v14 = *v7;
    if ( !v12 )
      goto LABEL_19;
    v15 = *(_QWORD **)(a2 + 8);
    a4 = *(unsigned int *)(a2 + 20);
    a3 = (__int64)&v15[a4];
    if ( v15 != (_QWORD *)a3 )
    {
      while ( v14 != *v15 )
      {
        if ( (_QWORD *)a3 == ++v15 )
          goto LABEL_50;
      }
      goto LABEL_8;
    }
LABEL_50:
    if ( (unsigned int)a4 < *(_DWORD *)(a2 + 16) )
    {
      a4 = (unsigned int)(a4 + 1);
      *(_DWORD *)(a2 + 20) = a4;
      *(_QWORD *)a3 = v14;
      v21 = *(_QWORD **)(a2 + 8);
      ++*(_QWORD *)a2;
      a3 = *(unsigned __int8 *)(a2 + 28);
    }
    else
    {
LABEL_19:
      sub_C8CC70(a2, v14, a3, a4, a5, a6);
      v21 = *(_QWORD **)(a2 + 8);
      a5 = v22;
      a3 = *(unsigned __int8 *)(a2 + 28);
      if ( !(_BYTE)a5 )
        goto LABEL_8;
    }
    v23 = v7[1];
    if ( !(_BYTE)a3 )
      goto LABEL_26;
    a4 = *(unsigned int *)(a2 + 20);
    a3 = (__int64)&v21[a4];
    if ( (_QWORD *)a3 != v21 )
    {
      while ( v23 != *v21 )
      {
        if ( (_QWORD *)a3 == ++v21 )
          goto LABEL_52;
      }
LABEL_25:
      ++v7;
      goto LABEL_8;
    }
LABEL_52:
    if ( (unsigned int)a4 < *(_DWORD *)(a2 + 16) )
    {
      a4 = (unsigned int)(a4 + 1);
      *(_DWORD *)(a2 + 20) = a4;
      *(_QWORD *)a3 = v23;
      v24 = *(_QWORD **)(a2 + 8);
      ++*(_QWORD *)a2;
      a3 = *(unsigned __int8 *)(a2 + 28);
    }
    else
    {
LABEL_26:
      sub_C8CC70(a2, v23, a3, a4, a5, a6);
      v24 = *(_QWORD **)(a2 + 8);
      a5 = v25;
      a3 = *(unsigned __int8 *)(a2 + 28);
      if ( !(_BYTE)a5 )
        goto LABEL_25;
    }
    v26 = v7[2];
    if ( !(_BYTE)a3 )
      goto LABEL_33;
    a4 = *(unsigned int *)(a2 + 20);
    a3 = (__int64)&v24[a4];
    if ( v24 != (_QWORD *)a3 )
    {
      while ( v26 != *v24 )
      {
        if ( (_QWORD *)a3 == ++v24 )
          goto LABEL_54;
      }
LABEL_32:
      v7 += 2;
      goto LABEL_8;
    }
LABEL_54:
    if ( (unsigned int)a4 < *(_DWORD *)(a2 + 16) )
    {
      a4 = (unsigned int)(a4 + 1);
      *(_DWORD *)(a2 + 20) = a4;
      *(_QWORD *)a3 = v26;
      v27 = *(_QWORD **)(a2 + 8);
      ++*(_QWORD *)a2;
      a3 = *(unsigned __int8 *)(a2 + 28);
    }
    else
    {
LABEL_33:
      sub_C8CC70(a2, v26, a3, a4, a5, a6);
      v27 = *(_QWORD **)(a2 + 8);
      a5 = v28;
      a3 = *(unsigned __int8 *)(a2 + 28);
      if ( !(_BYTE)a5 )
        goto LABEL_32;
    }
    v29 = v7[3];
    if ( !(_BYTE)a3 )
    {
LABEL_40:
      sub_C8CC70(a2, v29, a3, a4, a5, a6);
      v12 = *(_BYTE *)(a2 + 28);
      if ( !(_BYTE)a3 )
        goto LABEL_39;
      goto LABEL_41;
    }
    a4 = *(unsigned int *)(a2 + 20);
    a3 = (__int64)&v27[a4];
    if ( (_QWORD *)a3 != v27 )
      break;
LABEL_56:
    if ( (unsigned int)a4 >= *(_DWORD *)(a2 + 16) )
      goto LABEL_40;
    a4 = (unsigned int)(a4 + 1);
    *(_DWORD *)(a2 + 20) = a4;
    *(_QWORD *)a3 = v29;
    v12 = *(_BYTE *)(a2 + 28);
    ++*(_QWORD *)a2;
LABEL_41:
    v7 += 4;
    if ( v13 == v7 )
    {
      v10 = v9 - v7;
      goto LABEL_43;
    }
  }
  while ( v29 != *v27 )
  {
    if ( (_QWORD *)a3 == ++v27 )
      goto LABEL_56;
  }
LABEL_39:
  v7 += 3;
LABEL_8:
  if ( v9 != v7 )
  {
    for ( i = v7 + 1; v9 != i; *(v7 - 1) = v30 )
    {
LABEL_10:
      v17 = *i;
      if ( !*(_BYTE *)(a2 + 28) )
        goto LABEL_47;
      v18 = *(_QWORD **)(a2 + 8);
      a4 = *(unsigned int *)(a2 + 20);
      a3 = (__int64)&v18[a4];
      if ( v18 != (_QWORD *)a3 )
      {
        while ( v17 != *v18 )
        {
          if ( (_QWORD *)a3 == ++v18 )
            goto LABEL_58;
        }
LABEL_15:
        if ( v9 == ++i )
          break;
        goto LABEL_10;
      }
LABEL_58:
      if ( (unsigned int)a4 < *(_DWORD *)(a2 + 16) )
      {
        a4 = (unsigned int)(a4 + 1);
        *(_DWORD *)(a2 + 20) = a4;
        *(_QWORD *)a3 = v17;
        ++*(_QWORD *)a2;
      }
      else
      {
LABEL_47:
        sub_C8CC70(a2, v17, a3, a4, a5, a6);
        if ( !(_BYTE)a3 )
          goto LABEL_15;
      }
      v30 = *i++;
      ++v7;
    }
  }
LABEL_16:
  result = *(__int64 **)a1;
  v20 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v9;
  if ( v9 != (__int64 *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)) )
  {
    memmove(v7, v9, *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v9);
    result = *(__int64 **)a1;
  }
  *(_DWORD *)(a1 + 8) = ((char *)v7 + v20 - (char *)result) >> 3;
  return result;
}
