// Function: sub_3206D90
// Address: 0x3206d90
//
__int64 __fastcall sub_3206D90(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // r8d
  int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // r10d
  unsigned int i; // eax
  __int64 v11; // rsi
  unsigned int v12; // eax
  __int64 result; // rax
  unsigned int v14; // eax
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  unsigned int v19; // [rsp+Ch] [rbp-24h]

  v3 = 0x100000;
  v6 = *(_DWORD *)(a3 + 20);
  if ( (v6 & 0x2000) == 0 )
  {
    v3 = *(_DWORD *)(a3 + 20) & 0x4000;
    if ( (v6 & 0x4000) != 0 )
      v3 = 0x200000;
  }
  v7 = *(unsigned int *)(a1 + 1240);
  v8 = *(_QWORD *)(a1 + 1224);
  if ( (_DWORD)v7 )
  {
    v9 = 1;
    for ( i = (v7 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v7 - 1) & v12 )
    {
      v11 = v8 + 24LL * i;
      if ( a2 == *(_QWORD *)v11 && a3 == *(_QWORD *)(v11 + 8) )
        break;
      if ( *(_QWORD *)v11 == -4096 && *(_QWORD *)(v11 + 8) == -4096 )
        goto LABEL_13;
      v12 = v9 + i;
      ++v9;
    }
    if ( v11 != v8 + 24 * v7 )
      return *(unsigned int *)(v11 + 16);
  }
LABEL_13:
  ++*(_DWORD *)(a1 + 1328);
  v14 = sub_3206C30(a1, a2, v3);
  result = sub_31FEC80(a1, a2, v14, a3);
  v18 = *(unsigned int *)(a1 + 1328);
  if ( (_DWORD)v18 == 1 )
  {
    v19 = result;
    sub_32053F0(a1, a2, v18, v15, v16, v17);
    LODWORD(v18) = *(_DWORD *)(a1 + 1328);
    result = v19;
  }
  *(_DWORD *)(a1 + 1328) = v18 - 1;
  return result;
}
