// Function: sub_1E99220
// Address: 0x1e99220
//
__int64 __fastcall sub_1E99220(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 *v6; // rax
  char v7; // dl
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v11; // rdx
  __int64 *v12; // rdi
  __int64 *v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rax
  int v16; // eax
  int v17; // eax

  v5 = *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v6 = *(__int64 **)(a3 + 8);
  if ( *(__int64 **)(a3 + 16) != v6 )
    goto LABEL_2;
  v11 = *(unsigned int *)(a3 + 28);
  v12 = &v6[v11];
  if ( v6 != v12 )
  {
    v13 = 0;
    while ( a2 != *v6 )
    {
      if ( *v6 == -2 )
        v13 = v6;
      if ( v12 == ++v6 )
      {
        if ( !v13 )
          goto LABEL_32;
        *v13 = a2;
        v16 = *(_DWORD *)(a3 + 28);
        --*(_DWORD *)(a3 + 32);
        v17 = v16 - *(_DWORD *)(a3 + 32);
        ++*(_QWORD *)a3;
        if ( v17 != 16 )
          goto LABEL_4;
        return 0;
      }
    }
    return 1;
  }
LABEL_32:
  if ( (unsigned int)v11 < *(_DWORD *)(a3 + 24) )
  {
    *(_DWORD *)(a3 + 28) = v11 + 1;
    *v12 = a2;
    ++*(_QWORD *)a3;
  }
  else
  {
LABEL_2:
    sub_16CCBA0(a3, a2);
    if ( !v7 )
      return 1;
  }
  if ( *(_DWORD *)(a3 + 28) - *(_DWORD *)(a3 + 32) != 16 )
  {
LABEL_4:
    v8 = *(_QWORD *)(a1 + 232);
    if ( (int)v5 < 0 )
      v9 = *(_QWORD *)(*(_QWORD *)(v8 + 24) + 16 * (v5 & 0x7FFFFFFF) + 8);
    else
      v9 = *(_QWORD *)(*(_QWORD *)(v8 + 272) + 8 * v5);
    while ( v9 )
    {
      if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 && (*(_BYTE *)(v9 + 4) & 8) == 0 )
      {
        v14 = *(_QWORD *)(v9 + 16);
LABEL_20:
        if ( (!**(_WORD **)(v14 + 16) || **(_WORD **)(v14 + 16) == 45) && (unsigned __int8)sub_1E99220(a1, v14, a3) )
        {
          v15 = *(_QWORD *)(v9 + 16);
          while ( 1 )
          {
            v9 = *(_QWORD *)(v9 + 32);
            if ( !v9 )
              return 1;
            if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 && (*(_BYTE *)(v9 + 4) & 8) == 0 )
            {
              v14 = *(_QWORD *)(v9 + 16);
              if ( v15 != v14 )
                goto LABEL_20;
            }
          }
        }
        return 0;
      }
      v9 = *(_QWORD *)(v9 + 32);
    }
    return 1;
  }
  return 0;
}
