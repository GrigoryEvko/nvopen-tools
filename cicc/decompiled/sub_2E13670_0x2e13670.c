// Function: sub_2E13670
// Address: 0x2e13670
//
__int64 __fastcall sub_2E13670(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 *v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rsi
  __int64 v19; // rdx
  _QWORD *v20; // r10
  unsigned int v21; // r11d
  __int64 v22; // r9
  __int64 *v23; // rdi
  __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h]

  v5 = *(_QWORD *)(a2 + 64);
  v24 = v5 + 8LL * *(unsigned int *)(a2 + 72);
  if ( v5 == v24 )
    return 0;
  while ( 1 )
  {
    v6 = *(_QWORD *)(*(_QWORD *)v5 + 8LL);
    v7 = v6 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v6 & 0xFFFFFFFFFFFFFFF8LL) != 0 && (v6 & 6) == 0 )
    {
      v8 = *(_QWORD *)(v7 + 16);
      v9 = *(_QWORD *)(a1 + 32);
      if ( v8 )
      {
        v10 = *(_QWORD *)(v8 + 24);
      }
      else
      {
        v19 = *(unsigned int *)(v9 + 304);
        v20 = *(_QWORD **)(v9 + 296);
        if ( *(_DWORD *)(v9 + 304) )
        {
          v21 = *(_DWORD *)(v7 + 24);
          do
          {
            while ( 1 )
            {
              v22 = v19 >> 1;
              v23 = &v20[2 * (v19 >> 1)];
              if ( v21 < (*(_DWORD *)((*v23 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v23 >> 1) & 3) )
                break;
              v20 = v23 + 2;
              v19 = v19 - v22 - 1;
              if ( v19 <= 0 )
                goto LABEL_27;
            }
            v19 >>= 1;
          }
          while ( v22 > 0 );
        }
LABEL_27:
        v10 = *(v20 - 1);
      }
      v11 = *(unsigned int *)(v10 + 72);
      if ( (unsigned int)v11 > 0x64 )
        return 1;
      v12 = *(_QWORD *)(v10 + 64);
      v25 = v12 + 8 * v11;
      if ( v12 != v25 )
        break;
    }
LABEL_3:
    v5 += 8;
    if ( v24 == v5 )
      return 0;
  }
  v13 = v12;
  while ( 1 )
  {
    v17 = *(_QWORD *)(*(_QWORD *)(v9 + 152) + 16LL * *(unsigned int *)(*(_QWORD *)v13 + 24LL) + 8);
    v14 = ((v17 >> 1) & 3) != 0
        ? v17 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v17 >> 1) & 3) - 1))
        : *(_QWORD *)(v17 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
    v15 = (__int64 *)sub_2E09D00((__int64 *)a2, v14);
    v16 = 0;
    if ( v15 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
      && (*(_DWORD *)((*v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v15 >> 1) & 3) <= (*(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(v14 >> 1)
                                                                                             & 3) )
    {
      v16 = v15[2];
    }
    if ( a3 == v16 )
      return 1;
    v13 += 8;
    if ( v25 == v13 )
      goto LABEL_3;
    v9 = *(_QWORD *)(a1 + 32);
  }
}
