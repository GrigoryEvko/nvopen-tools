// Function: sub_3707D20
// Address: 0x3707d20
//
_QWORD *__fastcall sub_3707D20(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r12
  __int64 v5; // r15
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rsi
  _QWORD *i; // rdx
  __int64 v12; // rcx
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // edi
  unsigned int v17; // r10d
  __int64 v18; // r8
  __int64 v19; // r9
  int v20; // edx
  __int64 v21; // rcx
  _QWORD *j; // rdx
  int v23; // [rsp+Ch] [rbp-54h]
  __int64 v24; // [rsp+18h] [rbp-48h]

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
  result = (_QWORD *)sub_C7D670(12LL * v6, 1);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = unk_504EE80;
    v10 = v5 + 12 * v4;
    for ( i = (_QWORD *)((char *)result + 12 * v8); i != result; result = (_QWORD *)((char *)result + 12) )
    {
      if ( result )
        *result = v9;
    }
    v12 = unk_504EE80;
    v13 = qword_504EE78;
    if ( v10 != v5 )
    {
      v14 = v5;
      do
      {
        while ( 1 )
        {
          v15 = *(_QWORD *)v14;
          if ( *(_QWORD *)v14 != v12 && v15 != v13 )
            break;
          v14 += 12;
          if ( v10 == v14 )
            return (_QWORD *)sub_C7D6A0(v5, 12 * v4, 1);
        }
        v16 = *(_DWORD *)(a1 + 24);
        if ( !v16 )
        {
          MEMORY[0] = *(_QWORD *)v14;
          BUG();
        }
        v23 = 1;
        v24 = 0;
        v17 = (v16 - 1) & *(_DWORD *)v14;
        while ( 1 )
        {
          v18 = *(_QWORD *)(a1 + 8) + 12LL * v17;
          v19 = *(_QWORD *)v18;
          if ( *(_QWORD *)v18 == v15 )
            break;
          if ( unk_504EE80 == v19 )
          {
            if ( v24 )
              v18 = v24;
            break;
          }
          if ( v19 == qword_504EE78 )
          {
            if ( v24 )
              v18 = v24;
            v24 = v18;
          }
          v17 = (v16 - 1) & (v23 + v17);
          ++v23;
        }
        *(_QWORD *)v18 = v15;
        v20 = *(_DWORD *)(v14 + 8);
        v14 += 12;
        *(_DWORD *)(v18 + 8) = v20;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v10 != v14 );
    }
    return (_QWORD *)sub_C7D6A0(v5, 12 * v4, 1);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    v21 = unk_504EE80;
    for ( j = (_QWORD *)((char *)result + 12 * *(unsigned int *)(a1 + 24));
          j != result;
          result = (_QWORD *)((char *)result + 12) )
    {
      if ( result )
        *result = v21;
    }
  }
  return result;
}
