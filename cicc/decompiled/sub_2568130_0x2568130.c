// Function: sub_2568130
// Address: 0x2568130
//
__int64 **__fastcall sub_2568130(__int64 a1, __int64 *a2)
{
  int v2; // ebx
  __int64 v3; // r13
  __int64 v5; // r9
  int v6; // edx
  int v7; // ebx
  int v8; // r15d
  unsigned int i; // edx
  __int64 **v10; // r14
  __int64 *v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rcx
  char v14; // al
  __int64 v16; // rdi
  __int64 v17; // rsi
  unsigned __int64 v18; // rax
  unsigned int v19; // edx
  char v20; // al
  unsigned int v21; // [rsp+Ch] [rbp-34h]
  unsigned int v22; // [rsp+Ch] [rbp-34h]

  v2 = *(_DWORD *)(a1 + 24);
  if ( v2 )
  {
    v3 = *(_QWORD *)(a1 + 8);
    if ( !byte_4FEF240[0] && (unsigned int)sub_2207590((__int64)byte_4FEF240) )
    {
      unk_4FEF260 = -4096;
      unk_4FEF268 = -4096;
      qword_4FEF270 = 0;
      unk_4FEF278 = 0;
      sub_2207640((__int64)byte_4FEF240);
    }
    v5 = *a2;
    v6 = *(_DWORD *)(*a2 + 28);
    if ( !v6 )
    {
      v16 = *(_QWORD *)(v5 + 16);
      v17 = 0;
      if ( v16 )
        v17 = (unsigned int)sub_253B7A0(v16);
      v18 = ((0xBF58476D1CE4E5B9LL
            * (v17
             | (((0xBF58476D1CE4E5B9LL
                * (((unsigned int)*(_QWORD *)(v5 + 8) >> 9) ^ ((unsigned int)*(_QWORD *)(v5 + 8) >> 4)
                 | ((unsigned __int64)(((unsigned int)*(_QWORD *)v5 >> 9) ^ ((unsigned int)*(_QWORD *)v5 >> 4)) << 32)))
               ^ ((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)*(_QWORD *)(v5 + 8) >> 9) ^ ((unsigned int)*(_QWORD *)(v5 + 8) >> 4)
                  | ((unsigned __int64)(((unsigned int)*(_QWORD *)v5 >> 9) ^ ((unsigned int)*(_QWORD *)v5 >> 4)) << 32))) >> 31)) << 32))) >> 31)
          ^ (0xBF58476D1CE4E5B9LL
           * (v17
            | (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)*(_QWORD *)(v5 + 8) >> 9) ^ ((unsigned int)*(_QWORD *)(v5 + 8) >> 4)
                | ((unsigned __int64)(((unsigned int)*(_QWORD *)v5 >> 9) ^ ((unsigned int)*(_QWORD *)v5 >> 4)) << 32)))
              ^ ((0xBF58476D1CE4E5B9LL
                * (((unsigned int)*(_QWORD *)(v5 + 8) >> 9) ^ ((unsigned int)*(_QWORD *)(v5 + 8) >> 4)
                 | ((unsigned __int64)(((unsigned int)*(_QWORD *)v5 >> 9) ^ ((unsigned int)*(_QWORD *)v5 >> 4)) << 32))) >> 31)) << 32)));
      *(_DWORD *)(v5 + 28) = ((0xBF58476D1CE4E5B9LL
                             * (v17
                              | (((0xBF58476D1CE4E5B9LL
                                 * (((unsigned int)*(_QWORD *)(v5 + 8) >> 9) ^ ((unsigned int)*(_QWORD *)(v5 + 8) >> 4)
                                  | ((unsigned __int64)(((unsigned int)*(_QWORD *)v5 >> 9)
                                                      ^ ((unsigned int)*(_QWORD *)v5 >> 4)) << 32)))
                                ^ ((0xBF58476D1CE4E5B9LL
                                  * (((unsigned int)*(_QWORD *)(v5 + 8) >> 9) ^ ((unsigned int)*(_QWORD *)(v5 + 8) >> 4)
                                   | ((unsigned __int64)(((unsigned int)*(_QWORD *)v5 >> 9)
                                                       ^ ((unsigned int)*(_QWORD *)v5 >> 4)) << 32))) >> 31)) << 32))) >> 31)
                           ^ (484763065 * v17);
      v6 = v18;
      v5 = *a2;
    }
    v7 = v2 - 1;
    v8 = 1;
    for ( i = v7 & v6; ; i = v7 & v19 )
    {
      v10 = (__int64 **)(v3 + 8LL * i);
      v11 = *v10;
      v12 = **v10;
      v13 = (*v10)[1];
      if ( *(_QWORD *)(v5 + 8) == v13 && *(_QWORD *)v5 == v12 )
      {
        v22 = i;
        v20 = sub_254C7C0(*(__int64 **)(v5 + 16), v11[2]);
        i = v22;
        if ( v20 )
          return v10;
        v11 = *v10;
        v12 = **v10;
        v13 = (*v10)[1];
      }
      if ( unk_4FEF260 == v12 && unk_4FEF268 == v13 )
      {
        v21 = i;
        v14 = sub_254C7C0((__int64 *)v11[2], qword_4FEF270);
        i = v21;
        if ( v14 )
          break;
      }
      v19 = v8 + i;
      v5 = *a2;
      ++v8;
    }
  }
  return 0;
}
