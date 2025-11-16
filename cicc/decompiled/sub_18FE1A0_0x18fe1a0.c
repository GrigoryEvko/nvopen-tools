// Function: sub_18FE1A0
// Address: 0x18fe1a0
//
_QWORD *__fastcall sub_18FE1A0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r15
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r12
  _QWORD *i; // rdx
  __int64 *v10; // r14
  __int64 v11; // rdi
  int v12; // ebx
  __int64 *v13; // rbx
  __int64 v14; // rdx
  _QWORD *j; // rdx
  int v16; // [rsp+4h] [rbp-4Ch]
  __int64 *v17; // [rsp+8h] [rbp-48h]
  __int64 v18; // [rsp+10h] [rbp-40h]
  int v19; // [rsp+18h] [rbp-38h]
  unsigned int v20; // [rsp+1Ch] [rbp-34h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(16LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[2 * v3];
    for ( i = &result[2 * v7]; i != result; result += 2 )
    {
      if ( result )
        *result = -8;
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        v11 = *v10;
        if ( *v10 != -16 && v11 != -8 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *v10;
            BUG();
          }
          v18 = *(_QWORD *)(a1 + 8);
          v19 = 1;
          v16 = v12 - 1;
          v20 = (v12 - 1) & sub_18FDEE0(v11);
          v17 = 0;
          while ( 1 )
          {
            v13 = (__int64 *)(v18 + 16LL * v20);
            if ( (unsigned __int8)sub_18FB980(*v10, *v13) )
              break;
            if ( *v13 == -8 )
              goto LABEL_26;
            if ( *v13 == -16 )
            {
              if ( *v13 == -8 )
              {
LABEL_26:
                if ( v17 )
                  v13 = v17;
                break;
              }
              if ( !v17 )
              {
                if ( *v13 != -16 )
                  v13 = 0;
                v17 = v13;
              }
            }
            v20 = v16 & (v19 + v20);
            ++v19;
          }
          *v13 = *v10;
          v13[1] = v10[1];
          ++*(_DWORD *)(a1 + 16);
        }
        v10 += 2;
      }
      while ( v8 != v10 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v14 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v14]; j != result; result += 2 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
