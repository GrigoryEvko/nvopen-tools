// Function: sub_BBADB0
// Address: 0xbbadb0
//
__int64 __fastcall sub_BBADB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  _QWORD *v7; // rax
  __int64 v8; // rcx
  _QWORD *v9; // rbx
  __int64 v10; // r12
  _QWORD *v11; // r13
  __int64 *v12; // rbx
  __int64 result; // rax
  __int64 *v14; // r12
  __int64 v15; // rsi
  _QWORD *v16; // rdi
  _QWORD *v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // rsi
  _QWORD *v20; // rax
  __int64 v21; // rsi
  _QWORD *i; // rdx
  _QWORD *v23; // rax
  __int64 *v24; // r12
  __int64 v25; // rsi
  _QWORD *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rdi
  __int64 v30; // [rsp+8h] [rbp-38h]

  if ( *(_DWORD *)(a2 + 72) != *(_DWORD *)(a2 + 68) )
    goto LABEL_2;
  if ( !*(_BYTE *)(a2 + 28) )
  {
    result = sub_C8CA60(a2, &unk_4F82400, a3, a4);
    if ( result )
      return result;
LABEL_2:
    v6 = *(unsigned __int8 *)(a1 + 28);
    if ( *(_DWORD *)(a1 + 68) != *(_DWORD *)(a1 + 72) )
      goto LABEL_3;
    if ( (_BYTE)v6 )
    {
      result = *(_QWORD *)(a1 + 8);
      v28 = result + 8LL * *(unsigned int *)(a1 + 20);
      if ( result != v28 )
      {
        while ( *(_UNKNOWN **)result != &unk_4F82400 )
        {
          result += 8;
          if ( v28 == result )
            goto LABEL_3;
        }
        goto LABEL_69;
      }
    }
    else
    {
      result = sub_C8CA60(a1, &unk_4F82400, v6, a4);
      if ( result )
      {
LABEL_69:
        if ( a1 != a2 )
          result = sub_C8CF80(a1, a1 + 32, 2, a2 + 32, a2);
        v29 = a1 + 48;
        if ( a2 + 48 != a1 + 48 )
          return sub_C8CF80(v29, a1 + 80, 2, a2 + 80, a2 + 48);
        return result;
      }
      v6 = *(unsigned __int8 *)(a1 + 28);
    }
LABEL_3:
    v7 = *(_QWORD **)(a2 + 56);
    if ( *(_BYTE *)(a2 + 76) )
      v8 = *(unsigned int *)(a2 + 68);
    else
      v8 = *(unsigned int *)(a2 + 64);
    v9 = &v7[v8];
    if ( v7 == v9 )
      goto LABEL_8;
    while ( 1 )
    {
      v10 = *v7;
      v11 = v7;
      if ( *v7 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v9 == ++v7 )
        goto LABEL_8;
    }
    if ( v9 == v7 )
    {
LABEL_8:
      v12 = *(__int64 **)(a1 + 8);
      if ( (_BYTE)v6 )
      {
LABEL_38:
        result = *(unsigned int *)(a1 + 20);
        v24 = &v12[result];
        if ( v12 != v24 )
        {
          v25 = *v12;
          if ( !*(_BYTE *)(a2 + 28) )
            goto LABEL_47;
LABEL_40:
          result = *(_QWORD *)(a2 + 8);
          v6 = result + 8LL * *(unsigned int *)(a2 + 20);
          if ( result == v6 )
          {
LABEL_48:
            result = *--v24;
            *v12 = result;
            --*(_DWORD *)(a1 + 20);
            ++*(_QWORD *)a1;
            goto LABEL_45;
          }
          while ( v25 != *(_QWORD *)result )
          {
            result += 8;
            if ( v6 == result )
              goto LABEL_48;
          }
          while ( 1 )
          {
            ++v12;
LABEL_45:
            if ( v24 == v12 )
              break;
            v25 = *v12;
            if ( *(_BYTE *)(a2 + 28) )
              goto LABEL_40;
LABEL_47:
            result = sub_C8CA60(a2, v25, v6, v8);
            if ( !result )
              goto LABEL_48;
          }
        }
        return result;
      }
    }
    else
    {
      v30 = a1 + 48;
      if ( !(_BYTE)v6 )
        goto LABEL_51;
LABEL_24:
      v16 = *(_QWORD **)(a1 + 8);
      v17 = &v16[*(unsigned int *)(a1 + 20)];
      v18 = v16;
      if ( v16 != v17 )
      {
        while ( *v18 != v10 )
        {
          if ( v17 == ++v18 )
            goto LABEL_29;
        }
        v19 = (unsigned int)(*(_DWORD *)(a1 + 20) - 1);
        *(_DWORD *)(a1 + 20) = v19;
        *v18 = v16[v19];
        ++*(_QWORD *)a1;
      }
LABEL_29:
      if ( *(_BYTE *)(a1 + 76) )
      {
LABEL_30:
        v20 = *(_QWORD **)(a1 + 56);
        v21 = *(unsigned int *)(a1 + 68);
        for ( i = &v20[v21]; i != v20; ++v20 )
        {
          if ( *v20 == v10 )
            goto LABEL_34;
        }
        if ( (unsigned int)v21 < *(_DWORD *)(a1 + 64) )
        {
          *(_DWORD *)(a1 + 68) = v21 + 1;
          *i = v10;
          ++*(_QWORD *)(a1 + 48);
          goto LABEL_34;
        }
      }
      while ( 1 )
      {
        sub_C8CC70(v30, v10);
LABEL_34:
        v23 = v11 + 1;
        if ( v11 + 1 == v9 )
          break;
        while ( 1 )
        {
          v10 = *v23;
          v11 = v23;
          if ( *v23 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v9 == ++v23 )
            goto LABEL_37;
        }
        v6 = *(unsigned __int8 *)(a1 + 28);
        if ( v9 == v23 )
          goto LABEL_8;
        if ( (_BYTE)v6 )
          goto LABEL_24;
LABEL_51:
        v26 = (_QWORD *)sub_C8CA60(a1, v10, v6, v8);
        if ( !v26 )
          goto LABEL_29;
        *v26 = -2;
        ++*(_DWORD *)(a1 + 24);
        ++*(_QWORD *)a1;
        if ( *(_BYTE *)(a1 + 76) )
          goto LABEL_30;
      }
LABEL_37:
      v6 = *(unsigned __int8 *)(a1 + 28);
      v12 = *(__int64 **)(a1 + 8);
      if ( (_BYTE)v6 )
        goto LABEL_38;
    }
    result = *(unsigned int *)(a1 + 16);
    v14 = &v12[result];
    if ( v12 == v14 )
      return result;
    while ( 1 )
    {
      v15 = *v12;
      if ( (unsigned __int64)*v12 >= 0xFFFFFFFFFFFFFFFELL )
        goto LABEL_16;
      if ( *(_BYTE *)(a2 + 28) )
      {
        result = *(_QWORD *)(a2 + 8);
        v6 = result + 8LL * *(unsigned int *)(a2 + 20);
        if ( result == v6 )
          goto LABEL_20;
        while ( v15 != *(_QWORD *)result )
        {
          result += 8;
          if ( v6 == result )
            goto LABEL_20;
        }
LABEL_16:
        if ( ++v12 == v14 )
          return result;
      }
      else
      {
        result = sub_C8CA60(a2, v15, v6, v8);
        if ( result )
          goto LABEL_16;
LABEL_20:
        *v12++ = -2;
        ++*(_DWORD *)(a1 + 24);
        ++*(_QWORD *)a1;
        if ( v12 == v14 )
          return result;
      }
    }
  }
  result = *(_QWORD *)(a2 + 8);
  v27 = result + 8LL * *(unsigned int *)(a2 + 20);
  if ( result == v27 )
    goto LABEL_2;
  while ( *(_UNKNOWN **)result != &unk_4F82400 )
  {
    result += 8;
    if ( v27 == result )
      goto LABEL_2;
  }
  return result;
}
