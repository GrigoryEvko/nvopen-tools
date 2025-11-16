// Function: sub_1E70770
// Address: 0x1e70770
//
__int64 __fastcall sub_1E70770(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rbx
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // rdx
  __int64 v9; // rdx

  v4 = *(_QWORD *)(a1 + 56);
  v5 = *(_QWORD *)(a1 + 48);
  if ( v5 != v4 )
  {
    while ( 1 )
    {
      sub_1F02110(v5);
      if ( *(_DWORD *)(v5 + 208) )
      {
        if ( *(_DWORD *)(v5 + 212) )
          goto LABEL_4;
LABEL_9:
        v9 = *(unsigned int *)(a3 + 8);
        if ( (unsigned int)v9 >= *(_DWORD *)(a3 + 12) )
        {
          sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v6, v7);
          v9 = *(unsigned int *)(a3 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v5;
        v5 += 272;
        ++*(_DWORD *)(a3 + 8);
        if ( v4 == v5 )
          return sub_1F02110(a1 + 344);
      }
      else
      {
        v8 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v8 >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v6, v7);
          v8 = *(unsigned int *)(a2 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v8) = v5;
        ++*(_DWORD *)(a2 + 8);
        if ( !*(_DWORD *)(v5 + 212) )
          goto LABEL_9;
LABEL_4:
        v5 += 272;
        if ( v4 == v5 )
          return sub_1F02110(a1 + 344);
      }
    }
  }
  return sub_1F02110(a1 + 344);
}
