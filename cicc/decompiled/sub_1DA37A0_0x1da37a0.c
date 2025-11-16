// Function: sub_1DA37A0
// Address: 0x1da37a0
//
_QWORD *__fastcall sub_1DA37A0(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v3; // rax
  _QWORD *result; // rax
  __int64 v5; // rdx
  _QWORD *i; // rdx
  _QWORD *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdi
  int v10; // esi
  int v11; // r10d
  _QWORD *v12; // r9
  unsigned int v13; // ecx
  _QWORD *v14; // r14
  __int64 v15; // r8
  _QWORD *v16; // r12
  _QWORD *v17; // r15
  __int64 v18; // r13
  __int64 v19; // rax
  _QWORD *v20; // r12
  _QWORD *v21; // rdi
  int v22; // edx
  int v25; // [rsp+14h] [rbp-3Ch]
  __int64 v26; // [rsp+18h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 8) & 1LL;
  *(_QWORD *)(a1 + 8) = (unsigned int)v3;
  if ( (_DWORD)v3 )
  {
    result = (_QWORD *)(a1 + 16);
    v5 = 20;
  }
  else
  {
    result = *(_QWORD **)(a1 + 16);
    v5 = 5LL * *(unsigned int *)(a1 + 24);
  }
  for ( i = &result[v5]; i != result; result += 5 )
  {
    if ( result )
      *result = -8;
  }
  v7 = a2 + 2;
  if ( a2 != a3 )
  {
    while ( 1 )
    {
      v8 = *(v7 - 2);
      if ( v8 != -8 && v8 != -16 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v9 = a1 + 16;
          v10 = 3;
        }
        else
        {
          v9 = *(_QWORD *)(a1 + 16);
          v22 = *(_DWORD *)(a1 + 24);
          if ( !v22 )
          {
            MEMORY[0] = *(v7 - 2);
            BUG();
          }
          v10 = v22 - 1;
        }
        v11 = 1;
        v12 = 0;
        v13 = v10 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v14 = (_QWORD *)(v9 + 40LL * v13);
        v15 = *v14;
        if ( v8 != *v14 )
        {
          while ( v15 != -8 )
          {
            if ( v15 == -16 && !v12 )
              v12 = v14;
            v13 = v10 & (v11 + v13);
            v14 = (_QWORD *)(v9 + 40LL * v13);
            v15 = *v14;
            if ( v8 == *v14 )
              goto LABEL_13;
            ++v11;
          }
          if ( v12 )
            v14 = v12;
        }
LABEL_13:
        v16 = v14 + 2;
        *v14 = v8;
        v14[1] = 0;
        v14[3] = v14 + 2;
        v14[2] = v14 + 2;
        v14[4] = 0;
        v17 = (_QWORD *)*v7;
        if ( (_QWORD *)*v7 != v7 )
        {
          do
          {
            v18 = v17[4];
            v25 = *((_DWORD *)v17 + 4);
            v26 = v17[3];
            v19 = sub_22077B0(40);
            *(_QWORD *)(v19 + 32) = v18;
            *(_QWORD *)(v19 + 24) = v26;
            *(_DWORD *)(v19 + 16) = v25;
            sub_2208C80(v19, v14 + 2);
            ++v14[4];
            v17 = (_QWORD *)*v17;
          }
          while ( v17 != v7 );
          v16 = (_QWORD *)v14[2];
        }
        v14[1] = v16;
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v20 = (_QWORD *)*v7;
        while ( v20 != v7 )
        {
          v21 = v20;
          v20 = (_QWORD *)*v20;
          j_j___libc_free_0(v21, 40);
        }
      }
      result = v7 + 5;
      if ( a3 == v7 + 3 )
        break;
      v7 += 5;
    }
  }
  return result;
}
