// Function: sub_37C5570
// Address: 0x37c5570
//
_QWORD *__fastcall sub_37C5570(__int64 a1, int a2)
{
  unsigned int v3; // r12d
  __int64 v4; // r15
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r14
  __int64 v10; // r8
  _QWORD *i; // rdx
  __int64 v12; // r12
  __int64 v13; // r9
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  int v16; // edi
  int v17; // edi
  __int64 v18; // r11
  unsigned __int64 v19; // rcx
  unsigned int v20; // esi
  __int64 *v21; // rcx
  __int64 v22; // r10
  __int64 v23; // rcx
  _QWORD *j; // rdx
  __int64 *v25; // rdx
  int v26; // [rsp+14h] [rbp-4Ch]
  __int64 *v27; // [rsp+18h] [rbp-48h]

  v3 = *(_DWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(16LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = unk_5051170;
    v9 = 16LL * v3;
    v10 = v4 + v9;
    for ( i = &result[2 * v7]; i != result; result += 2 )
    {
      if ( result )
        *result = v8;
    }
    v12 = unk_5051170;
    v13 = qword_5051168;
    if ( v10 != v4 )
    {
      v14 = v4;
      do
      {
        v15 = *(_QWORD *)v14;
        if ( v13 != *(_QWORD *)v14 && v12 != v15 )
        {
          v16 = *(_DWORD *)(a1 + 24);
          if ( !v16 )
          {
            MEMORY[0] = *(_QWORD *)v14;
            BUG();
          }
          v17 = v16 - 1;
          v18 = *(_QWORD *)(a1 + 8);
          v19 = 0x9DDFEA08EB382D69LL * (HIDWORD(v15) ^ (((8 * v15) & 0x7FFFFFFF8LL) + 12995744));
          v20 = v17
              & (-348639895
               * (((unsigned int)((0x9DDFEA08EB382D69LL * ((v19 >> 47) ^ v19 ^ HIDWORD(v15))) >> 32) >> 15)
                ^ (-348639895 * ((v19 >> 47) ^ v19 ^ HIDWORD(v15)))));
          v21 = (__int64 *)(v18 + 16LL * v20);
          v22 = *v21;
          if ( v15 != *v21 )
          {
            v26 = 1;
            v27 = (__int64 *)(v18 + 16LL * v20);
            v21 = 0;
            while ( unk_5051170 != v22 )
            {
              if ( v21 || qword_5051168 != v22 )
                v27 = v21;
              v20 = v17 & (v26 + v20);
              v21 = (__int64 *)(v18 + 16LL * v20);
              v22 = *v21;
              if ( *(_QWORD *)v14 == *v21 )
                goto LABEL_14;
              ++v26;
              v25 = v27;
              v27 = (__int64 *)(v18 + 16LL * v20);
              v21 = v25;
            }
            if ( !v21 )
              v21 = v27;
          }
LABEL_14:
          *v21 = *(_QWORD *)v14;
          *((_DWORD *)v21 + 2) = *(_DWORD *)(v14 + 8);
          ++*(_DWORD *)(a1 + 16);
        }
        v14 += 16;
      }
      while ( v10 != v14 );
    }
    return (_QWORD *)sub_C7D6A0(v4, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    v23 = unk_5051170;
    for ( j = &result[2 * *(unsigned int *)(a1 + 24)]; j != result; result += 2 )
    {
      if ( result )
        *result = v23;
    }
  }
  return result;
}
