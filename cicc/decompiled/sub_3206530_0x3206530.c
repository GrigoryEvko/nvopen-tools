// Function: sub_3206530
// Address: 0x3206530
//
__int64 __fastcall sub_3206530(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // rcx
  int v7; // r9d
  unsigned int i; // eax
  __int64 v9; // rsi
  unsigned int v10; // eax
  __int64 result; // rax
  unsigned int v12; // eax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  unsigned int v17; // [rsp-3Ch] [rbp-3Ch]

  if ( !a2 )
    return 3;
  v5 = *(unsigned int *)(a1 + 1240);
  v6 = *(_QWORD *)(a1 + 1224);
  if ( (_DWORD)v5 )
  {
    v7 = 1;
    for ( i = (v5 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v5 - 1) & v10 )
    {
      v9 = v6 + 24LL * i;
      if ( a2 == *(unsigned __int8 **)v9 && a3 == *(_QWORD *)(v9 + 8) )
        break;
      if ( *(_QWORD *)v9 == -4096 && *(_QWORD *)(v9 + 8) == -4096 )
        goto LABEL_9;
      v10 = v7 + i;
      ++v7;
    }
    if ( v9 != v6 + 24 * v5 )
      return *(unsigned int *)(v9 + 16);
  }
LABEL_9:
  ++*(_DWORD *)(a1 + 1328);
  v12 = sub_3206290(a1, a2, a3);
  result = sub_31FEC80(a1, (__int64)a2, v12, a3);
  v16 = *(unsigned int *)(a1 + 1328);
  if ( (_DWORD)v16 == 1 )
  {
    v17 = result;
    sub_32053F0(a1, (__int64)a2, v16, v13, v14, v15);
    LODWORD(v16) = *(_DWORD *)(a1 + 1328);
    result = v17;
  }
  *(_DWORD *)(a1 + 1328) = v16 - 1;
  return result;
}
