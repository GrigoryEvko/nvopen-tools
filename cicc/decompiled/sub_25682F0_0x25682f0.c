// Function: sub_25682F0
// Address: 0x25682f0
//
char __fastcall sub_25682F0(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // r14d
  char result; // al
  __int64 v6; // r12
  __int64 v8; // r9
  int v9; // edx
  int v10; // ecx
  int v11; // r10d
  __int64 *v12; // r8
  unsigned int i; // r15d
  __int64 *v14; // r14
  _QWORD *v15; // rdi
  char v16; // al
  __int64 *v17; // r8
  __int64 v18; // rdi
  __int64 v19; // rsi
  unsigned __int64 v20; // rax
  char v21; // al
  __int64 *v22; // [rsp+0h] [rbp-40h]
  __int64 *v23; // [rsp+0h] [rbp-40h]
  int v24; // [rsp+8h] [rbp-38h]
  int v25; // [rsp+8h] [rbp-38h]
  int v26; // [rsp+Ch] [rbp-34h]
  int v27; // [rsp+Ch] [rbp-34h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    if ( !byte_4FEF240[0] && (unsigned int)sub_2207590((__int64)byte_4FEF240) )
    {
      qword_4FEF260 = -4096;
      unk_4FEF268 = -4096;
      qword_4FEF270 = 0;
      unk_4FEF278 = 0;
      sub_2207640((__int64)byte_4FEF240);
    }
    if ( !byte_4FEF208[0] && (unsigned int)sub_2207590((__int64)byte_4FEF208) )
    {
      qword_4FEF220 = -8192;
      unk_4FEF228 = -8192;
      qword_4FEF230 = 0;
      unk_4FEF238 = 0;
      sub_2207640((__int64)byte_4FEF208);
    }
    v8 = *a2;
    v9 = *(_DWORD *)(*a2 + 28);
    if ( !v9 )
    {
      v18 = *(_QWORD *)(v8 + 16);
      v19 = 0;
      if ( v18 )
        v19 = (unsigned int)sub_253B7A0(v18);
      v20 = ((0xBF58476D1CE4E5B9LL
            * (v19
             | (((0xBF58476D1CE4E5B9LL
                * (((unsigned int)*(_QWORD *)(v8 + 8) >> 9) ^ ((unsigned int)*(_QWORD *)(v8 + 8) >> 4)
                 | ((unsigned __int64)(((unsigned int)*(_QWORD *)v8 >> 9) ^ ((unsigned int)*(_QWORD *)v8 >> 4)) << 32)))
               ^ ((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)*(_QWORD *)(v8 + 8) >> 9) ^ ((unsigned int)*(_QWORD *)(v8 + 8) >> 4)
                  | ((unsigned __int64)(((unsigned int)*(_QWORD *)v8 >> 9) ^ ((unsigned int)*(_QWORD *)v8 >> 4)) << 32))) >> 31)) << 32))) >> 31)
          ^ (0xBF58476D1CE4E5B9LL
           * (v19
            | (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)*(_QWORD *)(v8 + 8) >> 9) ^ ((unsigned int)*(_QWORD *)(v8 + 8) >> 4)
                | ((unsigned __int64)(((unsigned int)*(_QWORD *)v8 >> 9) ^ ((unsigned int)*(_QWORD *)v8 >> 4)) << 32)))
              ^ ((0xBF58476D1CE4E5B9LL
                * (((unsigned int)*(_QWORD *)(v8 + 8) >> 9) ^ ((unsigned int)*(_QWORD *)(v8 + 8) >> 4)
                 | ((unsigned __int64)(((unsigned int)*(_QWORD *)v8 >> 9) ^ ((unsigned int)*(_QWORD *)v8 >> 4)) << 32))) >> 31)) << 32)));
      *(_DWORD *)(v8 + 28) = ((0xBF58476D1CE4E5B9LL
                             * (v19
                              | (((0xBF58476D1CE4E5B9LL
                                 * (((unsigned int)*(_QWORD *)(v8 + 8) >> 9) ^ ((unsigned int)*(_QWORD *)(v8 + 8) >> 4)
                                  | ((unsigned __int64)(((unsigned int)*(_QWORD *)v8 >> 9)
                                                      ^ ((unsigned int)*(_QWORD *)v8 >> 4)) << 32)))
                                ^ ((0xBF58476D1CE4E5B9LL
                                  * (((unsigned int)*(_QWORD *)(v8 + 8) >> 9) ^ ((unsigned int)*(_QWORD *)(v8 + 8) >> 4)
                                   | ((unsigned __int64)(((unsigned int)*(_QWORD *)v8 >> 9)
                                                       ^ ((unsigned int)*(_QWORD *)v8 >> 4)) << 32))) >> 31)) << 32))) >> 31)
                           ^ (484763065 * v19);
      v9 = v20;
      v8 = *a2;
    }
    v10 = v4 - 1;
    v11 = 1;
    v12 = 0;
    for ( i = (v4 - 1) & v9; ; i = v26 & (i + v24) )
    {
      v14 = (__int64 *)(v6 + 8LL * i);
      v15 = (_QWORD *)*v14;
      if ( *(_QWORD *)(*v14 + 8) == *(_QWORD *)(v8 + 8) && *v15 == *(_QWORD *)v8 )
      {
        v25 = v11;
        v23 = v12;
        v27 = v10;
        result = sub_254C7C0(*(__int64 **)(v8 + 16), v15[2]);
        v10 = v27;
        v12 = v23;
        v11 = v25;
        if ( result )
        {
          *a3 = v14;
          return result;
        }
        v15 = (_QWORD *)*v14;
      }
      v24 = v11;
      v22 = v12;
      v26 = v10;
      v16 = sub_2561100((__int64)v15, &qword_4FEF260);
      v17 = v22;
      if ( v16 )
        break;
      v21 = sub_2561100(*v14, &qword_4FEF220);
      v12 = v22;
      v10 = v26;
      if ( !v22 && v21 )
        v12 = (__int64 *)(v6 + 8LL * i);
      v8 = *a2;
      v11 = v24 + 1;
    }
    if ( !v22 )
      v17 = (__int64 *)(v6 + 8LL * i);
    *a3 = v17;
    return 0;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
