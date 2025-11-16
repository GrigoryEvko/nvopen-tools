// Function: sub_28D54E0
// Address: 0x28d54e0
//
_QWORD *__fastcall sub_28D54E0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rdx
  _QWORD *v8; // r13
  _QWORD *i; // rdx
  _QWORD *v10; // r15
  __int64 v11; // rdx
  int v12; // ecx
  int v13; // edi
  __int64 v14; // r8
  int v15; // r11d
  _QWORD *v16; // r10
  unsigned int v17; // esi
  _QWORD *v18; // rcx
  __int64 v19; // r9
  _QWORD *v20; // rdi
  _QWORD *v21; // rdx
  _QWORD *v22; // rsi
  _QWORD *v23; // rbx
  unsigned __int64 v24; // rdi
  _QWORD *j; // rdx
  __int64 v26; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(40LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v26 = 40 * v3;
    v8 = (_QWORD *)(v4 + 40 * v3);
    for ( i = &result[5 * v7]; i != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
    v10 = (_QWORD *)(v4 + 8);
    if ( v8 != (_QWORD *)v4 )
    {
      while ( 1 )
      {
        v11 = *(v10 - 1);
        if ( v11 != -8192 && v11 != -4096 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *(v10 - 1);
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = (_QWORD *)(v14 + 40LL * v17);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -4096 )
            {
              if ( !v16 && v19 == -8192 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = (_QWORD *)(v14 + 40LL * v17);
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_13;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_13:
          *v18 = v11;
          v20 = (_QWORD *)*v10;
          v21 = v18 + 1;
          v18[1] = *v10;
          v22 = (_QWORD *)v10[1];
          v18[2] = v22;
          v18[3] = v10[2];
          if ( v10 == v20 )
          {
            v18[2] = v21;
            v18[1] = v21;
          }
          else
          {
            *v22 = v21;
            *(_QWORD *)(v18[1] + 8LL) = v21;
            v10[1] = v10;
            *v10 = v10;
            v10[2] = 0;
            v21 = (_QWORD *)v18[1];
          }
          v18[4] = v21;
          ++*(_DWORD *)(a1 + 16);
          v23 = (_QWORD *)*v10;
          while ( v10 != v23 )
          {
            v24 = (unsigned __int64)v23;
            v23 = (_QWORD *)*v23;
            j_j___libc_free_0(v24);
          }
        }
        if ( v8 == v10 + 4 )
          break;
        v10 += 5;
      }
    }
    return (_QWORD *)sub_C7D6A0(v4, v26, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[5 * *(unsigned int *)(a1 + 24)]; j != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
