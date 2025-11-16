// Function: sub_1857B40
// Address: 0x1857b40
//
unsigned __int64 *__fastcall sub_1857B40(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int64 *result; // rax
  __int64 v5; // rdx
  unsigned __int64 *v6; // rdi
  unsigned int v7; // r8d
  unsigned __int64 *v8; // rcx
  __int64 v9; // rdx
  unsigned __int64 v11; // rdi
  _QWORD *v12; // r9
  _QWORD *v13; // rax
  _QWORD *v14; // r8
  _QWORD *v15; // rdx
  __int64 v16; // rdx
  unsigned __int64 *v17; // r13
  __int64 v18; // rsi
  unsigned __int64 *v19; // rbx
  __int64 *v20; // r8
  __int64 *v21; // r9
  __int64 *v22; // rdi
  unsigned int v23; // r10d
  __int64 *v24; // rax
  __int64 *v25; // rcx
  __int64 v26; // r14
  __int64 i; // r13
  _QWORD *v28; // rax
  __int64 v29; // rdx
  unsigned __int64 *v30; // r13
  __int64 v31; // rsi
  unsigned __int64 *v32; // rbx
  __int64 *v33; // r8
  __int64 *v34; // r9
  __int64 *v35; // rdi
  unsigned int v36; // r10d
  __int64 *v37; // rax
  __int64 *v38; // rcx
  unsigned __int64 v39[5]; // [rsp+8h] [rbp-28h] BYREF

  result = (unsigned __int64 *)*(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)result > 0x17u )
  {
    a2 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL);
    result = *(unsigned __int64 **)(a3 + 8);
    if ( *(unsigned __int64 **)(a3 + 16) != result )
      return sub_16CCBA0(a3, a2);
    v5 = *(unsigned int *)(a3 + 28);
    v6 = &result[v5];
    v7 = v5;
    if ( result != v6 )
    {
      v8 = 0;
      while ( a2 != *result )
      {
        if ( *result == -2 )
          v8 = result;
        if ( v6 == ++result )
          goto LABEL_20;
      }
      return result;
    }
LABEL_35:
    if ( v7 >= *(_DWORD *)(a3 + 24) )
      return sub_16CCBA0(a3, a2);
    *(_DWORD *)(a3 + 28) = v7 + 1;
    *v6 = a2;
    ++*(_QWORD *)a3;
    return result;
  }
  if ( (unsigned __int8)result <= 3u )
  {
    result = *(unsigned __int64 **)(a3 + 8);
    if ( *(unsigned __int64 **)(a3 + 16) != result )
      return sub_16CCBA0(a3, a2);
    v9 = *(unsigned int *)(a3 + 28);
    v6 = &result[v9];
    v7 = v9;
    if ( result != v6 )
    {
      v8 = 0;
      while ( a2 != *result )
      {
        if ( *result == -2 )
          v8 = result;
        if ( v6 == ++result )
        {
LABEL_20:
          if ( !v8 )
            goto LABEL_35;
          *v8 = a2;
          --*(_DWORD *)(a3 + 32);
          ++*(_QWORD *)a3;
          return result;
        }
      }
      return result;
    }
    goto LABEL_35;
  }
  if ( (unsigned __int8)result > 0x10u )
    return result;
  v11 = *(_QWORD *)(a1 + 336);
  v39[0] = a2;
  v12 = *(_QWORD **)(*(_QWORD *)(a1 + 328) + 8 * (a2 % v11));
  if ( v12 )
  {
    v13 = (_QWORD *)*v12;
    if ( a2 == *(_QWORD *)(*v12 + 8LL) )
    {
LABEL_28:
      v15 = (_QWORD *)*v12;
      if ( *v12 )
      {
        result = (unsigned __int64 *)v15[4];
        if ( result == (unsigned __int64 *)v15[3] )
          v16 = *((unsigned int *)v15 + 11);
        else
          v16 = *((unsigned int *)v15 + 10);
        v17 = &result[v16];
        if ( result != v17 )
        {
          while ( 1 )
          {
            v18 = *result;
            v19 = result;
            if ( *result < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v17 == ++result )
              return result;
          }
          if ( result != v17 )
          {
            v20 = *(__int64 **)(a3 + 16);
            v21 = *(__int64 **)(a3 + 8);
            if ( v20 == v21 )
              goto LABEL_46;
LABEL_39:
            sub_16CCBA0(a3, v18);
            v20 = *(__int64 **)(a3 + 16);
            v21 = *(__int64 **)(a3 + 8);
LABEL_40:
            while ( 1 )
            {
              result = v19 + 1;
              if ( v19 + 1 == v17 )
                break;
              v18 = *result;
              for ( ++v19; *result >= 0xFFFFFFFFFFFFFFFELL; v19 = result )
              {
                if ( v17 == ++result )
                  return result;
                v18 = *result;
              }
              if ( v19 == v17 )
                return result;
              if ( v20 != v21 )
                goto LABEL_39;
LABEL_46:
              v22 = &v20[*(unsigned int *)(a3 + 28)];
              v23 = *(_DWORD *)(a3 + 28);
              if ( v22 == v20 )
              {
LABEL_62:
                if ( v23 >= *(_DWORD *)(a3 + 24) )
                  goto LABEL_39;
                *(_DWORD *)(a3 + 28) = v23 + 1;
                *v22 = v18;
                v21 = *(__int64 **)(a3 + 8);
                ++*(_QWORD *)a3;
                v20 = *(__int64 **)(a3 + 16);
              }
              else
              {
                v24 = v20;
                v25 = 0;
                while ( *v24 != v18 )
                {
                  if ( *v24 == -2 )
                    v25 = v24;
                  if ( v22 == ++v24 )
                  {
                    if ( !v25 )
                      goto LABEL_62;
                    *v25 = v18;
                    v20 = *(__int64 **)(a3 + 16);
                    --*(_DWORD *)(a3 + 32);
                    v21 = *(__int64 **)(a3 + 8);
                    ++*(_QWORD *)a3;
                    goto LABEL_40;
                  }
                }
              }
            }
          }
        }
        return result;
      }
    }
    else
    {
      while ( 1 )
      {
        v14 = (_QWORD *)*v13;
        if ( !*v13 )
          break;
        v12 = v13;
        if ( a2 % v11 != v14[1] % v11 )
          break;
        v13 = (_QWORD *)*v13;
        if ( a2 == v14[1] )
          goto LABEL_28;
      }
    }
  }
  v26 = sub_18578B0((_QWORD *)(a1 + 328), v39);
  for ( i = *(_QWORD *)(v39[0] + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v28 = sub_1648700(i);
    sub_1857B40(a1, v28, v26);
  }
  result = *(unsigned __int64 **)(v26 + 16);
  if ( result == *(unsigned __int64 **)(v26 + 8) )
    v29 = *(unsigned int *)(v26 + 28);
  else
    v29 = *(unsigned int *)(v26 + 24);
  v30 = &result[v29];
  if ( result != v30 )
  {
    while ( 1 )
    {
      v31 = *result;
      v32 = result;
      if ( *result < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v30 == ++result )
        return result;
    }
    if ( result != v30 )
    {
      v33 = *(__int64 **)(a3 + 16);
      v34 = *(__int64 **)(a3 + 8);
      while ( 1 )
      {
        if ( v33 != v34 )
          goto LABEL_67;
        v35 = &v33[*(unsigned int *)(a3 + 28)];
        v36 = *(_DWORD *)(a3 + 28);
        if ( v35 != v33 )
        {
          v37 = v33;
          v38 = 0;
          while ( *v37 != v31 )
          {
            if ( *v37 == -2 )
              v38 = v37;
            if ( v35 == ++v37 )
            {
              if ( !v38 )
                goto LABEL_83;
              *v38 = v31;
              v33 = *(__int64 **)(a3 + 16);
              --*(_DWORD *)(a3 + 32);
              v34 = *(__int64 **)(a3 + 8);
              ++*(_QWORD *)a3;
              goto LABEL_68;
            }
          }
          goto LABEL_68;
        }
LABEL_83:
        if ( v36 < *(_DWORD *)(a3 + 24) )
        {
          *(_DWORD *)(a3 + 28) = v36 + 1;
          *v35 = v31;
          v34 = *(__int64 **)(a3 + 8);
          ++*(_QWORD *)a3;
          v33 = *(__int64 **)(a3 + 16);
        }
        else
        {
LABEL_67:
          sub_16CCBA0(a3, v31);
          v33 = *(__int64 **)(a3 + 16);
          v34 = *(__int64 **)(a3 + 8);
        }
LABEL_68:
        result = v32 + 1;
        if ( v32 + 1 != v30 )
        {
          while ( 1 )
          {
            v31 = *result;
            v32 = result;
            if ( *result < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v30 == ++result )
              return result;
          }
          if ( result != v30 )
            continue;
        }
        return result;
      }
    }
  }
  return result;
}
