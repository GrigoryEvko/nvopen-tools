// Function: sub_3214F00
// Address: 0x3214f00
//
__int64 __fastcall sub_3214F00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r14
  char v9; // dl
  __int16 v10; // cx
  const void *v11; // rsi
  _QWORD *v12; // rax
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // r13
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // r13
  unsigned __int64 *v20; // rax
  __int64 v21; // rbx
  unsigned __int64 v22; // r13
  unsigned __int64 *v23; // rax
  unsigned __int64 v25; // [rsp+0h] [rbp-50h]
  unsigned __int64 v26; // [rsp+8h] [rbp-48h]
  __int64 v27; // [rsp+8h] [rbp-48h]

  v9 = *(_BYTE *)(a2 + 30);
  if ( !v9 )
    v9 = *(_QWORD *)(a2 + 32) != 0;
  v10 = *(_WORD *)(a2 + 28);
  v11 = (const void *)(a1 + 32);
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_WORD *)(a1 + 12) = v10;
  *(_BYTE *)(a1 + 14) = v9;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  *(_QWORD *)(a1 + 24) = 0xC00000000LL;
  v12 = *(_QWORD **)(a2 + 8);
  if ( v12 )
  {
    v13 = *v12 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v13 )
    {
      v14 = 12;
      v15 = 0;
      v16 = a1 + 16;
      while ( 1 )
      {
        v17 = *(unsigned __int16 *)(v13 + 14);
        v18 = v15 + 1;
        if ( (_WORD)v17 == 33 )
        {
          LOWORD(a5) = *(_WORD *)(v13 + 12);
          a6 = *(_QWORD *)(v13 + 16);
          a5 = a5 & 0xFFFFFFFF0000FFFFLL | 0x210000;
          v22 = a5;
          if ( v18 > v14 )
          {
            v25 = a5;
            v27 = *(_QWORD *)(v13 + 16);
            sub_C8D5F0(v16, v11, v18, 0x10u, a5, a6);
            v15 = *(unsigned int *)(a1 + 24);
            a5 = v25;
            a6 = v27;
          }
          v23 = (unsigned __int64 *)(*(_QWORD *)(a1 + 16) + 16 * v15);
          *v23 = v22;
          v23[1] = a6;
          ++*(_DWORD *)(a1 + 24);
          v21 = *(_QWORD *)v13;
          if ( (v21 & 4) != 0 )
            return a1;
        }
        else
        {
          LOWORD(v6) = *(_WORD *)(v13 + 12);
          v19 = v6 & 0xFFFFFFFF0000FFFFLL | (v17 << 16);
          v6 = v19;
          if ( v18 > v14 )
          {
            v26 = a5;
            sub_C8D5F0(v16, v11, v18, 0x10u, a5, a6);
            v15 = *(unsigned int *)(a1 + 24);
            a5 = v26;
          }
          v20 = (unsigned __int64 *)(*(_QWORD *)(a1 + 16) + 16 * v15);
          *v20 = v19;
          v20[1] = 0;
          ++*(_DWORD *)(a1 + 24);
          v21 = *(_QWORD *)v13;
          if ( (v21 & 4) != 0 )
            return a1;
        }
        v13 = v21 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v13 )
          return a1;
        v15 = *(unsigned int *)(a1 + 24);
        v14 = *(unsigned int *)(a1 + 28);
      }
    }
  }
  return a1;
}
