// Function: sub_277D170
// Address: 0x277d170
//
_QWORD *__fastcall sub_277D170(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // ebx
  __int64 v5; // r15
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 *v10; // r13
  _QWORD *i; // rdx
  __int64 *v12; // r12
  __int64 *v13; // rdi
  int v14; // ebx
  int v15; // eax
  __int64 *v16; // rbx
  char v17; // al
  __int64 v18; // rdx
  _QWORD *j; // rdx
  int v20; // [rsp+Ch] [rbp-54h]
  __int64 *v21; // [rsp+10h] [rbp-50h]
  __int64 v22; // [rsp+18h] [rbp-48h]
  int v23; // [rsp+20h] [rbp-40h]
  unsigned int v24; // [rsp+24h] [rbp-3Ch]
  __int64 v25; // [rsp+28h] [rbp-38h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
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
  result = (_QWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 16LL * v4;
    v10 = (__int64 *)(v5 + v9);
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v10 != (__int64 *)v5 )
    {
      v12 = (__int64 *)v5;
      do
      {
        v13 = (__int64 *)*v12;
        if ( *v12 != -8192 && v13 != (__int64 *)-4096LL )
        {
          v14 = *(_DWORD *)(a1 + 24);
          v25 = v9;
          if ( !v14 )
          {
            MEMORY[0] = *v12;
            BUG();
          }
          v22 = *(_QWORD *)(a1 + 8);
          v15 = sub_277CF80(v13);
          v23 = 1;
          v9 = v25;
          v20 = v14 - 1;
          v24 = (v14 - 1) & v15;
          v21 = 0;
          while ( 1 )
          {
            v26 = v9;
            v16 = (__int64 *)(v22 + 16LL * v24);
            v17 = sub_27792F0(*v12, *v16);
            v9 = v26;
            if ( v17 )
              break;
            if ( *v16 == -4096 )
              goto LABEL_26;
            if ( *v16 == -8192 )
            {
              if ( *v16 == -4096 )
              {
LABEL_26:
                if ( v21 )
                  v16 = v21;
                break;
              }
              if ( !v21 )
              {
                if ( *v16 != -8192 )
                  v16 = 0;
                v21 = v16;
              }
            }
            v24 = v20 & (v23 + v24);
            ++v23;
          }
          *v16 = *v12;
          v16[1] = v12[1];
          ++*(_DWORD *)(a1 + 16);
        }
        v12 += 2;
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v18 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v18]; j != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
