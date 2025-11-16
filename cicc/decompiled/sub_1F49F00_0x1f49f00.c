// Function: sub_1F49F00
// Address: 0x1f49f00
//
__int64 __fastcall sub_1F49F00(
        __int64 a1,
        int a2,
        unsigned __int16 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v9; // r13
  __int64 v11; // rax
  int *v12; // rbx
  int *v13; // r14
  int *v14; // rax
  int v15; // r9d
  __int64 v16; // rsi
  unsigned __int16 *v17; // rax
  int v18; // r9d
  __int64 v19; // rax
  __int16 v21; // [rsp+4h] [rbp-4Ch]
  int v23[13]; // [rsp+1Ch] [rbp-34h] BYREF
  __int64 v24; // [rsp+60h] [rbp+10h]

  v9 = *(_QWORD *)(a6 + 40);
  v11 = *(_QWORD *)(v9 + 208) + 40LL * (a2 & 0x7FFFFFFF);
  v12 = *(int **)(v11 + 8);
  v13 = &v12[*(unsigned int *)(v11 + 16)];
  if ( v12 != v13 )
  {
    if ( !*(_DWORD *)v11 )
    {
      v15 = *v12;
      goto LABEL_5;
    }
    while ( 1 )
    {
      v14 = v12++;
      if ( v13 == v12 )
        break;
      while ( 1 )
      {
        v15 = v14[1];
LABEL_5:
        v23[0] = v15;
        if ( a7 && v15 < 0 )
        {
          v15 = *(_DWORD *)(*(_QWORD *)(a7 + 264) + 4LL * (v15 & 0x7FFFFFFF));
          v23[0] = v15;
        }
        if ( v15 <= 0 )
          break;
        if ( (*(_QWORD *)(*(_QWORD *)(v9 + 304) + 8LL * ((unsigned int)v15 >> 6)) & (1LL << v15)) != 0 )
          break;
        v24 = a7;
        v16 = (__int64)&a3[a4];
        v17 = sub_1F49E30(a3, v16, v23);
        a7 = v24;
        if ( (unsigned __int16 *)v16 == v17 )
          break;
        v19 = *(unsigned int *)(a5 + 8);
        if ( (unsigned int)v19 >= *(_DWORD *)(a5 + 12) )
        {
          v21 = v18;
          sub_16CD150(a5, (const void *)(a5 + 16), 0, 2, v24, v18);
          v19 = *(unsigned int *)(a5 + 8);
          a7 = v24;
          LOWORD(v18) = v21;
        }
        *(_WORD *)(*(_QWORD *)a5 + 2 * v19) = v18;
        v14 = v12++;
        ++*(_DWORD *)(a5 + 8);
        if ( v13 == v12 )
          return 0;
      }
    }
  }
  return 0;
}
