// Function: sub_3207400
// Address: 0x3207400
//
__int64 __fastcall sub_3207400(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  unsigned __int8 v4; // al
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // r9d
  unsigned int i; // eax
  __int64 v11; // rsi
  unsigned int v12; // eax
  __int64 result; // rax
  __int64 v14; // r15
  int v15; // r14d
  const void *v16; // rax
  size_t v17; // rdx
  unsigned __int8 v18; // cl
  __int64 v19; // rsi
  char v20; // r9
  unsigned __int8 v21; // al
  __int64 v22; // r15
  unsigned int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdx
  unsigned int v28; // [rsp+Ch] [rbp-34h]

  v4 = *(_BYTE *)(a2 - 16);
  if ( (v4 & 2) != 0 )
    v5 = *(_QWORD *)(a2 - 32);
  else
    v5 = a2 - 16 - 8LL * ((v4 >> 2) & 0xF);
  v6 = *(_QWORD *)(v5 + 48);
  v7 = *(unsigned int *)(a1 + 1240);
  v8 = *(_QWORD *)(a1 + 1224);
  if ( !v6 )
    v6 = a2;
  if ( (_DWORD)v7 )
  {
    v9 = 1;
    for ( i = (v7 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v7 - 1) & v12 )
    {
      v11 = v8 + 24LL * i;
      if ( v6 == *(_QWORD *)v11 && a3 == *(unsigned __int8 **)(v11 + 8) )
        break;
      if ( *(_QWORD *)v11 == -4096 && *(_QWORD *)(v11 + 8) == -4096 )
        goto LABEL_15;
      v12 = v9 + i;
      ++v9;
    }
    if ( v11 != v8 + 24 * v7 )
      return *(unsigned int *)(v11 + 16);
  }
LABEL_15:
  ++*(_DWORD *)(a1 + 1328);
  v14 = v6 - 16;
  v15 = (*(_DWORD *)(v6 + 32) >> 12) & 1;
  v16 = (const void *)sub_A547D0(v6, 2);
  v18 = *(_BYTE *)(v6 - 16);
  if ( (v18 & 2) != 0 )
    v19 = *(_QWORD *)(v6 - 32);
  else
    v19 = v14 - 8LL * ((v18 >> 2) & 0xF);
  v20 = sub_31F74D0(*(_QWORD *)(v19 + 32), (__int64)a3, v16, v17);
  v21 = *(_BYTE *)(v6 - 16);
  if ( (v21 & 2) != 0 )
    v22 = *(_QWORD *)(v6 - 32);
  else
    v22 = v14 - 8LL * ((v21 >> 2) & 0xF);
  v23 = sub_3206EF0(a1, *(_QWORD *)(v22 + 32), a3, *(_DWORD *)(v6 + 28), v15, v20);
  result = sub_31FEC80(a1, v6, v23, (__int64)a3);
  v27 = *(unsigned int *)(a1 + 1328);
  if ( (_DWORD)v27 == 1 )
  {
    v28 = result;
    sub_32053F0(a1, v6, v27, v24, v25, v26);
    LODWORD(v27) = *(_DWORD *)(a1 + 1328);
    result = v28;
  }
  *(_DWORD *)(a1 + 1328) = v27 - 1;
  return result;
}
