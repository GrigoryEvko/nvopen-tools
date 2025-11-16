// Function: sub_37BA470
// Address: 0x37ba470
//
_QWORD *__fastcall sub_37BA470(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned int a4,
        __int64 a5,
        unsigned __int64 a6)
{
  __int64 v7; // r14
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // r12
  __int64 v12; // rdx
  unsigned int v13; // esi
  _DWORD *v14; // rax
  _DWORD *v15; // rdi
  __int64 v16; // rax
  _QWORD *result; // rax
  int v18; // eax
  unsigned int v19; // edi
  __int64 v20; // rsi
  unsigned __int64 v21; // r11
  __int64 v22; // rax
  __int64 v23; // r10
  unsigned int v24; // eax
  __int64 v25; // [rsp+8h] [rbp-48h]
  unsigned int v26; // [rsp+14h] [rbp-3Ch]
  unsigned int *v27; // [rsp+18h] [rbp-38h]

  v7 = a4;
  v9 = *(unsigned int *)(a1 + 40);
  v10 = 4 * v9;
  if ( (_DWORD)v9 )
  {
    a5 = a3;
    v11 = 0;
    v12 = a1 + 168;
    do
    {
      v13 = *(_DWORD *)(*(_QWORD *)(a1 + 88) + v11);
      if ( *(_DWORD *)(a1 + 284) > v13 )
      {
        if ( !*(_QWORD *)(a1 + 200) )
        {
          v14 = *(_DWORD **)(a1 + 112);
          v15 = &v14[*(unsigned int *)(a1 + 120)];
          if ( v14 != v15 )
          {
            while ( v13 != *v14 )
            {
              if ( v15 == ++v14 )
                goto LABEL_16;
            }
            if ( v15 != v14 )
              goto LABEL_10;
          }
LABEL_16:
          v18 = *(_DWORD *)(*(_QWORD *)(a2 + 24) + 4LL * (v13 >> 5));
          if ( !_bittest(&v18, v13) )
          {
            v19 = *(_DWORD *)(*(_QWORD *)(a1 + 64) + 4LL * v13);
            if ( v19 == -1 )
            {
              v25 = v12;
              v26 = a5;
              v27 = (unsigned int *)(*(_QWORD *)(a1 + 64) + 4LL * v13);
              v24 = sub_37BA230(a1, v13);
              v12 = v25;
              a5 = v26;
              v19 = v24;
              *v27 = v24;
            }
            v20 = *(_QWORD *)(a1 + 32) + 8LL * v19;
            a6 = *(_QWORD *)v20 & 0xFFFFFF0000000000LL;
            v21 = a6 | a5 & 0xFFFFF | ((unsigned __int64)(v7 & 0xFFFFF) << 20);
            *(_QWORD *)v20 = v21;
            *(_DWORD *)(v20 + 4) = BYTE4(v21) | (v19 << 8);
          }
          goto LABEL_10;
        }
        v22 = *(_QWORD *)(a1 + 176);
        if ( !v22 )
          goto LABEL_16;
        v23 = v12;
        do
        {
          a6 = *(_QWORD *)(v22 + 16);
          if ( v13 > *(_DWORD *)(v22 + 32) )
          {
            v22 = *(_QWORD *)(v22 + 24);
          }
          else
          {
            v23 = v22;
            v22 = *(_QWORD *)(v22 + 16);
          }
        }
        while ( v22 );
        if ( v12 == v23 || v13 < *(_DWORD *)(v23 + 32) )
          goto LABEL_16;
      }
LABEL_10:
      v11 += 4;
    }
    while ( v10 != v11 );
  }
  v16 = *(unsigned int *)(a1 + 304);
  if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 308) )
  {
    sub_C8D5F0(a1 + 296, (const void *)(a1 + 312), v16 + 1, 0x10u, a5, a6);
    v16 = *(unsigned int *)(a1 + 304);
  }
  result = (_QWORD *)(*(_QWORD *)(a1 + 296) + 16 * v16);
  *result = a2;
  result[1] = v7;
  ++*(_DWORD *)(a1 + 304);
  return result;
}
