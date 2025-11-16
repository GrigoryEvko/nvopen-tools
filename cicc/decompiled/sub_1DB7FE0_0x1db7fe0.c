// Function: sub_1DB7FE0
// Address: 0x1db7fe0
//
_QWORD *__fastcall sub_1DB7FE0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // r12
  unsigned __int64 v6; // r15
  __int64 v7; // r14
  signed __int64 v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // rdi
  unsigned int v11; // ecx
  unsigned int v12; // eax
  __int64 v13; // r12
  unsigned int v14; // eax
  __int64 v16; // rdx
  unsigned __int64 v17; // r10
  __int64 v18; // r8
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // r9
  __int64 v22; // r14
  unsigned int v23; // edi
  __int64 v24; // rcx
  __int64 *v25; // rsi
  unsigned int v26; // eax
  __int64 *v27[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = *(_QWORD **)(a1 + 96);
  v27[0] = (__int64 *)a1;
  if ( v5 )
  {
    if ( !v5[5] )
      return 0;
    v6 = a3 & 0xFFFFFFFFFFFFFFF8LL;
    v7 = (a3 >> 1) & 3;
    v8 = v7 ? v6 | (2LL * ((int)v7 - 1)) : *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL | 6;
    v9 = (_QWORD *)v5[2];
    if ( v9 )
    {
      v10 = (__int64)(v5 + 1);
      v11 = *(_DWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v8 >> 1) & 3;
      do
      {
        while ( 1 )
        {
          v12 = *(_DWORD *)((v9[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)v9[4] >> 1) & 3;
          if ( v12 > v11
            || v12 >= v11
            && ((unsigned int)v7 | *(_DWORD *)(v6 + 24)) < (*(_DWORD *)((v9[5] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                          | (unsigned int)((__int64)v9[5] >> 1) & 3) )
          {
            break;
          }
          v9 = (_QWORD *)v9[3];
          if ( !v9 )
            goto LABEL_12;
        }
        v10 = (__int64)v9;
        v9 = (_QWORD *)v9[2];
      }
      while ( v9 );
LABEL_12:
      if ( v5 + 1 != (_QWORD *)v10
        && v11 >= (*(_DWORD *)((*(_QWORD *)(v10 + 32) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                 | (unsigned int)(*(__int64 *)(v10 + 32) >> 1) & 3) )
      {
        v10 = sub_220EF30(v10);
      }
    }
    else
    {
      v10 = (__int64)(v5 + 1);
    }
    if ( v5[3] == v10 )
      return 0;
    v13 = sub_220EFE0(v10);
    v14 = *(_DWORD *)((*(_QWORD *)(v13 + 40) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(v13 + 40) >> 1) & 3;
    if ( v14 <= (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3) )
    {
      return 0;
    }
    else
    {
      if ( v14 < (*(_DWORD *)(v6 + 24) | (unsigned int)v7) )
        sub_1DB7AE0((__int64)v27, v13, a3);
      return *(_QWORD **)(v13 + 48);
    }
  }
  else
  {
    v16 = *(unsigned int *)(a1 + 8);
    if ( (_DWORD)v16 )
    {
      v17 = a3 & 0xFFFFFFFFFFFFFFF8LL;
      v18 = (a3 >> 1) & 3;
      if ( ((a3 >> 1) & 3) != 0 )
      {
        v19 = v17 | (2LL * ((int)v18 - 1));
        v20 = a3 & 0xFFFFFFFFFFFFFFF8LL | (2LL * ((int)v18 - 1)) & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        v20 = *(_QWORD *)v17 & 0xFFFFFFFFFFFFFFF8LL;
        v19 = v20 | 6;
      }
      v21 = *(_QWORD *)a1;
      v22 = *(_QWORD *)a1;
      v23 = *(_DWORD *)(v20 + 24) | (v19 >> 1) & 3;
      do
      {
        while ( 1 )
        {
          v24 = v16 >> 1;
          v25 = (__int64 *)(v22 + 8 * ((v16 >> 1) + (v16 & 0xFFFFFFFFFFFFFFFELL)));
          if ( v23 < (*(_DWORD *)((*v25 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v25 >> 1) & 3) )
            break;
          v22 = (__int64)(v25 + 3);
          v16 = v16 - v24 - 1;
          if ( v16 <= 0 )
            goto LABEL_29;
        }
        v16 >>= 1;
      }
      while ( v24 > 0 );
LABEL_29:
      if ( v21 != v22 )
      {
        v26 = *(_DWORD *)((*(_QWORD *)(v22 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(v22 - 16) >> 1) & 3;
        if ( v26 > (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3) )
        {
          if ( v26 < (*(_DWORD *)(v17 + 24) | (unsigned int)v18) )
            sub_1DB37E0(v27, (_QWORD *)(v22 - 24), a3);
          return *(_QWORD **)(v22 - 8);
        }
      }
    }
  }
  return v5;
}
