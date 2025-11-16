// Function: sub_25D7390
// Address: 0x25d7390
//
__int64 *__fastcall sub_25D7390(_QWORD *a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v6; // r13
  __int64 v7; // r12
  __int64 *result; // rax
  unsigned __int64 v9; // rsi
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  _QWORD *v15; // r15
  __int64 v16; // rdx
  __int64 *v17; // r13
  __int64 v18; // rsi
  __int64 *v19; // rbx
  _QWORD *v20; // rax
  unsigned __int64 v21; // r14
  unsigned __int64 v22; // r9
  __int64 v23; // rax
  _QWORD *v24; // rdi
  _QWORD *v25; // rax
  _QWORD *v26; // rsi
  __int64 i; // r13
  char v28; // di
  _QWORD *v29; // rax
  unsigned __int64 v30; // rsi
  char v31; // al
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // r15
  _QWORD *v34; // rdx
  void *v35; // rax
  void *v36; // rax
  _QWORD *v37; // r8
  _QWORD *v38; // rsi
  _QWORD *v39; // rdi
  __int64 v40; // rdx
  _QWORD **v41; // rax
  unsigned __int64 v42; // rdi
  __int64 v43; // rax
  size_t n; // [rsp+8h] [rbp-38h]
  __int64 na; // [rsp+8h] [rbp-38h]
  size_t nb; // [rsp+8h] [rbp-38h]

  v6 = a2;
  v7 = a3;
  result = (__int64 *)*a2;
  if ( (unsigned __int8)result <= 0x1Cu )
  {
    if ( (unsigned __int8)result <= 3u )
    {
      if ( !*(_BYTE *)(a3 + 28) )
        return sub_C8CC70(v7, (__int64)a2, a3, a4, a5, a6);
      result = *(__int64 **)(a3 + 8);
      a4 = *(unsigned int *)(a3 + 20);
      a3 = (__int64)&result[a4];
      if ( result == (__int64 *)a3 )
      {
LABEL_18:
        if ( (unsigned int)a4 >= *(_DWORD *)(v7 + 16) )
          return sub_C8CC70(v7, (__int64)a2, a3, a4, a5, a6);
        *(_DWORD *)(v7 + 20) = a4 + 1;
        *(_QWORD *)a3 = a2;
        ++*(_QWORD *)v7;
      }
      else
      {
        while ( a2 != (unsigned __int8 *)*result )
        {
          if ( (__int64 *)a3 == ++result )
            goto LABEL_18;
        }
      }
      return result;
    }
    if ( (unsigned __int8)result > 0x15u )
      return result;
    v9 = a1[42];
    v11 = *(_QWORD *)(a1[41] + 8 * ((unsigned __int64)v6 % v9));
    v12 = (unsigned __int64)v6 % v9;
    if ( v11 )
    {
      v13 = *(_QWORD *)v11;
      v14 = *(_QWORD *)(*(_QWORD *)v11 + 8LL);
      if ( v6 == (unsigned __int8 *)v14 )
      {
LABEL_26:
        v15 = *(_QWORD **)v11;
        if ( *(_QWORD *)v11 )
          goto LABEL_27;
      }
      else
      {
        while ( *(_QWORD *)v13 )
        {
          v14 = *(_QWORD *)(*(_QWORD *)v13 + 8LL);
          v11 = v13;
          if ( v12 != v14 % v9 )
            break;
          v13 = *(_QWORD *)v13;
          if ( v6 == (unsigned __int8 *)v14 )
            goto LABEL_26;
        }
      }
    }
    v20 = (_QWORD *)sub_22077B0(0x70u);
    v21 = (unsigned __int64)v20;
    if ( v20 )
      *v20 = 0;
    v22 = a1[42];
    v20[1] = v6;
    v20[3] = v20 + 6;
    v23 = a1[41];
    *(_QWORD *)(v21 + 16) = 0;
    *(_QWORD *)(v21 + 32) = 8;
    *(_DWORD *)(v21 + 40) = 0;
    *(_BYTE *)(v21 + 44) = 1;
    v24 = *(_QWORD **)(v23 + 8 * ((unsigned __int64)v6 % v22));
    if ( v24 )
    {
      v25 = (_QWORD *)*v24;
      if ( v6 == *(unsigned __int8 **)(*v24 + 8LL) )
      {
LABEL_42:
        v15 = (_QWORD *)*v24;
        if ( *v24 )
        {
          j_j___libc_free_0(v21);
LABEL_44:
          for ( i = *((_QWORD *)v6 + 2); i; i = *(_QWORD *)(i + 8) )
            sub_25D7390(a1, *(_QWORD *)(i + 24), v15 + 2);
LABEL_27:
          result = (__int64 *)v15[3];
          if ( *((_BYTE *)v15 + 44) )
            v16 = *((unsigned int *)v15 + 9);
          else
            v16 = *((unsigned int *)v15 + 8);
          v17 = &result[v16];
          if ( result != v17 )
          {
            while ( 1 )
            {
              v18 = *result;
              v19 = result;
              if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v17 == ++result )
                return result;
            }
            if ( result != v17 )
            {
              v28 = *(_BYTE *)(v7 + 28);
              if ( !v28 )
                goto LABEL_59;
LABEL_49:
              v29 = *(_QWORD **)(v7 + 8);
              v14 = *(unsigned int *)(v7 + 20);
              v16 = (__int64)&v29[v14];
              if ( v29 == (_QWORD *)v16 )
              {
LABEL_60:
                if ( (unsigned int)v14 < *(_DWORD *)(v7 + 16) )
                {
                  v14 = (unsigned int)(v14 + 1);
                  *(_DWORD *)(v7 + 20) = v14;
                  *(_QWORD *)v16 = v18;
                  v28 = *(_BYTE *)(v7 + 28);
                  ++*(_QWORD *)v7;
                  goto LABEL_53;
                }
                goto LABEL_59;
              }
              while ( *v29 != v18 )
              {
                if ( (_QWORD *)v16 == ++v29 )
                  goto LABEL_60;
              }
LABEL_53:
              while ( 1 )
              {
                result = v19 + 1;
                if ( v19 + 1 == v17 )
                  break;
                v18 = *result;
                for ( ++v19; (unsigned __int64)*result >= 0xFFFFFFFFFFFFFFFELL; v19 = result )
                {
                  if ( v17 == ++result )
                    return result;
                  v18 = *result;
                }
                if ( v19 == v17 )
                  return result;
                if ( v28 )
                  goto LABEL_49;
LABEL_59:
                sub_C8CC70(v7, v18, v16, v14, v11, v12);
                v28 = *(_BYTE *)(v7 + 28);
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
          v26 = (_QWORD *)*v25;
          if ( !*v25 )
            break;
          v24 = v25;
          if ( (unsigned __int64)v6 % v22 != v26[1] % v22 )
            break;
          v25 = (_QWORD *)*v25;
          if ( v6 == (unsigned __int8 *)v26[1] )
            goto LABEL_42;
        }
      }
    }
    v30 = v22;
    n = 8 * ((unsigned __int64)v6 % v22);
    v31 = sub_222DA10((__int64)(a1 + 45), v22, a1[44], 1);
    v33 = v32;
    if ( v31 )
    {
      if ( v32 == 1 )
      {
        v37 = a1 + 47;
        a1[47] = 0;
        v14 = (unsigned __int64)(a1 + 47);
      }
      else
      {
        if ( v32 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(a1 + 45, v30, v32);
        na = 8 * v32;
        v35 = (void *)sub_22077B0(8 * v32);
        v36 = memset(v35, 0, na);
        v37 = a1 + 47;
        v14 = (unsigned __int64)v36;
      }
      v38 = (_QWORD *)a1[43];
      a1[43] = 0;
      if ( v38 )
      {
        v12 = 0;
        do
        {
          v39 = v38;
          v38 = (_QWORD *)*v38;
          v40 = v39[1] % v33;
          v41 = (_QWORD **)(v14 + 8 * v40);
          if ( *v41 )
          {
            *v39 = **v41;
            **v41 = v39;
          }
          else
          {
            *v39 = a1[43];
            a1[43] = v39;
            *v41 = a1 + 43;
            if ( *v39 )
              *(_QWORD *)(v14 + 8 * v12) = v39;
            v12 = v40;
          }
        }
        while ( v38 );
      }
      v42 = a1[41];
      if ( (_QWORD *)v42 != v37 )
      {
        nb = v14;
        j_j___libc_free_0(v42);
        v14 = nb;
      }
      a1[42] = v33;
      a1[41] = v14;
      v11 = 8 * ((unsigned __int64)v6 % v33);
    }
    else
    {
      v14 = a1[41];
      v11 = n;
    }
    v34 = *(_QWORD **)(v14 + v11);
    if ( v34 )
    {
      *(_QWORD *)v21 = *v34;
      **(_QWORD **)(v14 + v11) = v21;
    }
    else
    {
      v43 = a1[43];
      a1[43] = v21;
      *(_QWORD *)v21 = v43;
      if ( v43 )
        *(_QWORD *)(v14 + 8LL * (*(_QWORD *)(v43 + 8) % a1[42])) = v21;
      *(_QWORD *)(a1[41] + v11) = a1 + 43;
    }
    ++a1[44];
    v15 = (_QWORD *)v21;
    goto LABEL_44;
  }
  a2 = *(unsigned __int8 **)(*((_QWORD *)a2 + 5) + 72LL);
  if ( !*(_BYTE *)(a3 + 28) )
    return sub_C8CC70(v7, (__int64)a2, a3, a4, a5, a6);
  result = *(__int64 **)(a3 + 8);
  a4 = *(unsigned int *)(a3 + 20);
  a3 = (__int64)&result[a4];
  if ( result == (__int64 *)a3 )
  {
LABEL_8:
    if ( (unsigned int)a4 >= *(_DWORD *)(v7 + 16) )
      return sub_C8CC70(v7, (__int64)a2, a3, a4, a5, a6);
    *(_DWORD *)(v7 + 20) = a4 + 1;
    *(_QWORD *)a3 = a2;
    ++*(_QWORD *)v7;
  }
  else
  {
    while ( a2 != (unsigned __int8 *)*result )
    {
      if ( (__int64 *)a3 == ++result )
        goto LABEL_8;
    }
  }
  return result;
}
