// Function: sub_2A65CC0
// Address: 0x2a65cc0
//
unsigned __int64 *__fastcall sub_2A65CC0(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  int v5; // r9d
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // r12
  int v9; // ebx
  unsigned __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // rsi
  int v13; // r11d
  unsigned int i; // eax
  __int64 v15; // r14
  unsigned int v16; // eax
  unsigned __int8 v18; // al
  unsigned int v19; // eax
  unsigned int v20; // eax
  unsigned __int64 v21; // [rsp-50h] [rbp-50h]
  unsigned __int64 v22; // [rsp-50h] [rbp-50h]
  unsigned __int64 v23; // [rsp-50h] [rbp-50h]
  int v24; // [rsp-44h] [rbp-44h]
  int v25; // [rsp-44h] [rbp-44h]
  int v26; // [rsp-44h] [rbp-44h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v3 = *(_QWORD *)(a3 + 8);
  if ( *(_BYTE *)(v3 + 8) != 15 )
    BUG();
  v5 = *(_DWORD *)(v3 + 12);
  if ( v5 )
  {
    v7 = 0;
    v8 = 0;
    v9 = 0;
    v10 = (unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32;
    while ( 1 )
    {
      v11 = *(unsigned int *)(a2 + 192);
      v12 = *(_QWORD *)(a2 + 176);
      if ( (_DWORD)v11 )
      {
        v13 = 1;
        for ( i = (v11 - 1)
                & (((0xBF58476D1CE4E5B9LL * (v10 | (unsigned int)(37 * v9))) >> 31)
                 ^ (484763065 * (v10 | (37 * v9)))); ; i = (v11 - 1) & v16 )
        {
          v15 = v12 + 56LL * i;
          if ( a3 == *(_QWORD *)v15 && *(_DWORD *)(v15 + 8) == v9 )
            break;
          if ( *(_QWORD *)v15 == -4096 && *(_DWORD *)(v15 + 8) == -1 )
            goto LABEL_11;
          v16 = v13 + i;
          ++v13;
        }
        if ( v8 != v7 )
        {
LABEL_12:
          if ( v8 )
          {
            v18 = *(_BYTE *)(v15 + 16);
            *(_WORD *)v8 = v18;
            if ( v18 > 3u )
            {
              if ( (unsigned __int8)(v18 - 4) > 1u )
                goto LABEL_16;
              v19 = *(_DWORD *)(v15 + 32);
              *(_DWORD *)(v8 + 16) = v19;
              if ( v19 > 0x40 )
              {
                v23 = v10;
                v26 = v5;
                sub_C43780(v8 + 8, (const void **)(v15 + 24));
                v10 = v23;
                v5 = v26;
              }
              else
              {
                *(_QWORD *)(v8 + 8) = *(_QWORD *)(v15 + 24);
              }
              v20 = *(_DWORD *)(v15 + 48);
              *(_DWORD *)(v8 + 32) = v20;
              if ( v20 > 0x40 )
              {
                v22 = v10;
                v25 = v5;
                sub_C43780(v8 + 24, (const void **)(v15 + 40));
                v10 = v22;
                v5 = v25;
              }
              else
              {
                *(_QWORD *)(v8 + 24) = *(_QWORD *)(v15 + 40);
              }
              *(_BYTE *)(v8 + 1) = *(_BYTE *)(v15 + 17);
              v8 = a1[1];
            }
            else
            {
              if ( v18 > 1u )
                *(_QWORD *)(v8 + 8) = *(_QWORD *)(v15 + 24);
LABEL_16:
              v8 = a1[1];
            }
          }
          a1[1] = v8 + 40;
          goto LABEL_18;
        }
      }
      else
      {
LABEL_11:
        v15 = v12 + 56 * v11;
        if ( v8 != v7 )
          goto LABEL_12;
      }
      v21 = v10;
      v24 = v5;
      sub_2A65970(a1, (unsigned __int8 *)v8, (unsigned __int8 *)(v15 + 16));
      v10 = v21;
      v5 = v24;
LABEL_18:
      if ( ++v9 == v5 )
        return a1;
      v8 = a1[1];
      v7 = a1[2];
    }
  }
  return a1;
}
