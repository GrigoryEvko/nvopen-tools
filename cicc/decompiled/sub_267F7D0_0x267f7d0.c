// Function: sub_267F7D0
// Address: 0x267f7d0
//
_QWORD *__fastcall sub_267F7D0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 *v10; // r15
  _QWORD *i; // rdx
  __int64 *v12; // rbx
  __int64 v13; // rax
  int v14; // edx
  int v15; // esi
  __int64 v16; // rdi
  int v17; // r11d
  _QWORD *v18; // r10
  unsigned int v19; // ecx
  _QWORD *v20; // rdx
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rax
  volatile signed __int32 *v24; // rdi
  signed __int32 v25; // eax
  signed __int32 v26; // eax
  __int64 v27; // rdx
  _QWORD *j; // rdx

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_QWORD *)sub_C7D670(24LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 24 * v4;
    v10 = (__int64 *)(v5 + 24 * v4);
    for ( i = &result[3 * v8]; i != result; result += 3 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v10 != (__int64 *)v5 )
    {
      v12 = (__int64 *)v5;
      do
      {
        v13 = *v12;
        if ( *v12 != -8192 && v13 != -4096 )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = *v12;
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 1;
          v18 = 0;
          v19 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v20 = (_QWORD *)(v16 + 24LL * v19);
          v21 = *v20;
          if ( v13 != *v20 )
          {
            while ( v21 != -4096 )
            {
              if ( !v18 && v21 == -8192 )
                v18 = v20;
              v19 = v15 & (v17 + v19);
              v20 = (_QWORD *)(v16 + 24LL * v19);
              v21 = *v20;
              if ( v13 == *v20 )
                goto LABEL_14;
              ++v17;
            }
            if ( v18 )
              v20 = v18;
          }
LABEL_14:
          *v20 = v13;
          v22 = v12[1];
          v20[2] = 0;
          v20[1] = v22;
          v23 = v12[2];
          v12[2] = 0;
          v20[2] = v23;
          v12[1] = 0;
          ++*(_DWORD *)(a1 + 16);
          v24 = (volatile signed __int32 *)v12[2];
          if ( v24 )
          {
            if ( &_pthread_key_create )
            {
              v25 = _InterlockedExchangeAdd(v24 + 2, 0xFFFFFFFF);
            }
            else
            {
              v25 = *((_DWORD *)v24 + 2);
              *((_DWORD *)v24 + 2) = v25 - 1;
            }
            if ( v25 == 1 )
            {
              (*(void (**)(void))(*(_QWORD *)v24 + 16LL))();
              if ( &_pthread_key_create )
              {
                v26 = _InterlockedExchangeAdd(v24 + 3, 0xFFFFFFFF);
              }
              else
              {
                v26 = *((_DWORD *)v24 + 3);
                *((_DWORD *)v24 + 3) = v26 - 1;
              }
              if ( v26 == 1 )
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v24 + 24LL))(v24);
            }
          }
        }
        v12 += 3;
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v27 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[3 * v27]; j != result; result += 3 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
