// Function: sub_FAF280
// Address: 0xfaf280
//
_QWORD *__fastcall sub_FAF280(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 **v10; // r12
  _QWORD *i; // rdx
  __int64 **j; // r15
  char v13; // al
  __int64 **v14; // rcx
  __int64 *v15; // rax
  __int64 v16; // rdx
  _QWORD *k; // rdx
  char v18; // al
  char v19; // al
  __int64 v20; // rcx
  _BYTE v21[12]; // [rsp+Ch] [rbp-54h]
  __int64 **v22; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+20h] [rbp-40h]
  int v24; // [rsp+28h] [rbp-38h]
  int v25; // [rsp+2Ch] [rbp-34h]
  unsigned int v26; // [rsp+2Ch] [rbp-34h]

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
  result = (_QWORD *)sub_C7D670(8LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 8 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = (__int64 **)(v5 + v9);
    for ( i = &result[v8]; i != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = (__int64 **)v5; v10 != j; ++*(_DWORD *)(a1 + 16) )
    {
      while ( (unsigned __int8)sub_FAEA90(*j, (__int64 *)0xFFFFFFFFFFFFF000LL)
           || (unsigned __int8)sub_FAEA90(*j, (__int64 *)0xFFFFFFFFFFFFE000LL) )
      {
        if ( v10 == ++j )
          return (_QWORD *)sub_C7D6A0(v5, v9, 8);
      }
      v25 = *(_DWORD *)(a1 + 24);
      if ( !v25 )
      {
        MEMORY[0] = *j;
        BUG();
      }
      v23 = *(_QWORD *)(a1 + 8);
      v24 = 1;
      *(_DWORD *)&v21[8] = 0;
      *(_QWORD *)v21 = (unsigned int)(v25 - 1);
      v26 = (v25 - 1) & sub_FAE360(*j);
      while ( 1 )
      {
        v22 = (__int64 **)(v23 + 8LL * v26);
        v13 = sub_FAEA90(*j, *v22);
        v14 = v22;
        if ( v13 )
          break;
        v18 = sub_FAEA90(*v22, (__int64 *)0xFFFFFFFFFFFFF000LL);
        v14 = (__int64 **)(v23 + 8LL * v26);
        if ( v18 )
        {
          if ( *(_QWORD *)&v21[4] )
            v14 = *(__int64 ***)&v21[4];
          break;
        }
        v19 = sub_FAEA90(*v22, (__int64 *)0xFFFFFFFFFFFFE000LL);
        if ( !*(_QWORD *)&v21[4] )
        {
          v20 = v23 + 8LL * v26;
          if ( !v19 )
            v20 = 0;
          *(_QWORD *)&v21[4] = v20;
        }
        v26 = *(_DWORD *)v21 & (v24 + v26);
        ++v24;
      }
      v15 = *j++;
      *v14 = v15;
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v16 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[v16]; k != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
