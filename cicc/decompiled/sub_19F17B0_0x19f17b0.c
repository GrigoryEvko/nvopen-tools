// Function: sub_19F17B0
// Address: 0x19f17b0
//
_QWORD *__fastcall sub_19F17B0(__int64 a1, int a2)
{
  _QWORD *v2; // r14
  __int64 v3; // rbx
  unsigned int v4; // eax
  _QWORD *result; // rax
  _QWORD *i; // rdx
  _QWORD *v7; // rbx
  __int64 v8; // rax
  int v9; // edx
  __int64 v10; // r8
  int v11; // esi
  int v12; // r10d
  _QWORD *v13; // r9
  unsigned int v14; // edx
  _QWORD *v15; // r15
  __int64 v16; // rdi
  _QWORD *v17; // r12
  _QWORD *v18; // r14
  __int64 v19; // r13
  __int64 v20; // rax
  _QWORD *v21; // r12
  _QWORD *v22; // rdi
  __int64 v23; // rdx
  _QWORD *j; // rdx
  _QWORD *v25; // [rsp+8h] [rbp-58h]
  _QWORD *v27; // [rsp+18h] [rbp-48h]
  int v28; // [rsp+24h] [rbp-3Ch]
  __int64 v29; // [rsp+28h] [rbp-38h]

  v2 = *(_QWORD **)(a1 + 8);
  v3 = *(unsigned int *)(a1 + 24);
  v25 = v2;
  v4 = sub_1454B60((unsigned int)(a2 - 1));
  if ( v4 < 0x40 )
    v4 = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = (_QWORD *)sub_22077B0(40LL * v4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v2 )
  {
    v27 = &v2[5 * v3];
    *(_QWORD *)(a1 + 16) = 0;
    for ( i = &result[5 * *(unsigned int *)(a1 + 24)]; i != result; result += 5 )
    {
      if ( result )
        *result = -8;
    }
    v7 = v2 + 2;
    if ( v27 != v2 )
    {
      while ( 1 )
      {
        v8 = *(v7 - 2);
        if ( v8 != -16 && v8 != -8 )
        {
          v9 = *(_DWORD *)(a1 + 24);
          if ( !v9 )
          {
            MEMORY[0] = *(v7 - 2);
            BUG();
          }
          v10 = *(_QWORD *)(a1 + 8);
          v11 = v9 - 1;
          v12 = 1;
          v13 = 0;
          v14 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v15 = (_QWORD *)(v10 + 40LL * v14);
          v16 = *v15;
          if ( v8 != *v15 )
          {
            while ( v16 != -8 )
            {
              if ( v16 == -16 && !v13 )
                v13 = v15;
              v14 = v11 & (v12 + v14);
              v15 = (_QWORD *)(v10 + 40LL * v14);
              v16 = *v15;
              if ( v8 == *v15 )
                goto LABEL_13;
              ++v12;
            }
            if ( v13 )
              v15 = v13;
          }
LABEL_13:
          v17 = v15 + 2;
          *v15 = v8;
          v15[1] = 0;
          v15[3] = v15 + 2;
          v15[2] = v15 + 2;
          v15[4] = 0;
          v18 = (_QWORD *)*v7;
          if ( v7 != (_QWORD *)*v7 )
          {
            do
            {
              v19 = v18[4];
              v28 = *((_DWORD *)v18 + 4);
              v29 = v18[3];
              v20 = sub_22077B0(40);
              *(_QWORD *)(v20 + 32) = v19;
              *(_DWORD *)(v20 + 16) = v28;
              *(_QWORD *)(v20 + 24) = v29;
              sub_2208C80(v20, v15 + 2);
              ++v15[4];
              v18 = (_QWORD *)*v18;
            }
            while ( v7 != v18 );
            v17 = (_QWORD *)v15[2];
          }
          v15[1] = v17;
          ++*(_DWORD *)(a1 + 16);
          v21 = (_QWORD *)*v7;
          while ( v7 != v21 )
          {
            v22 = v21;
            v21 = (_QWORD *)*v21;
            j_j___libc_free_0(v22, 40);
          }
        }
        if ( v27 == v7 + 3 )
          break;
        v7 += 5;
      }
    }
    return (_QWORD *)j___libc_free_0(v25);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[5 * v23]; j != result; result += 5 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
