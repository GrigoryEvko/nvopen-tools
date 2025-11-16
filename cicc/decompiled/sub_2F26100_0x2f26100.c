// Function: sub_2F26100
// Address: 0x2f26100
//
__int64 __fastcall sub_2F26100(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  _QWORD *v8; // rax
  char v10; // dl
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // rsi
  __int64 v14; // rax
  int v15; // eax

  v6 = a3;
  v7 = *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  if ( *(_BYTE *)(a3 + 28) )
  {
    v8 = *(_QWORD **)(a3 + 8);
    a4 = *(unsigned int *)(a3 + 20);
    a3 = (__int64)&v8[a4];
    if ( v8 != (_QWORD *)a3 )
    {
      while ( a2 != *v8 )
      {
        if ( (_QWORD *)a3 == ++v8 )
          goto LABEL_7;
      }
      return 1;
    }
LABEL_7:
    if ( (unsigned int)a4 < *(_DWORD *)(v6 + 16) )
    {
      *(_DWORD *)(v6 + 20) = a4 + 1;
      *(_QWORD *)a3 = a2;
      v15 = *(_DWORD *)(v6 + 20) - *(_DWORD *)(v6 + 24);
      ++*(_QWORD *)v6;
      if ( v15 == 16 )
        return 0;
      goto LABEL_10;
    }
  }
  sub_C8CC70(v6, a2, a3, a4, a5, a6);
  if ( !v10 )
    return 1;
  if ( *(_DWORD *)(v6 + 20) - *(_DWORD *)(v6 + 24) != 16 )
  {
LABEL_10:
    v11 = *a1;
    if ( (int)v7 < 0 )
      v12 = *(_QWORD *)(*(_QWORD *)(v11 + 56) + 16 * (v7 & 0x7FFFFFFF) + 8);
    else
      v12 = *(_QWORD *)(*(_QWORD *)(v11 + 304) + 8 * v7);
    while ( 1 )
    {
      if ( !v12 )
        return 1;
      if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 && (*(_BYTE *)(v12 + 4) & 8) == 0 )
        break;
      v12 = *(_QWORD *)(v12 + 32);
    }
    v13 = *(_QWORD *)(v12 + 16);
LABEL_19:
    if ( (!*(_WORD *)(v13 + 68) || *(_WORD *)(v13 + 68) == 68) && (unsigned __int8)sub_2F26100(a1, v13, v6) )
    {
      v14 = *(_QWORD *)(v12 + 16);
      while ( 1 )
      {
        v12 = *(_QWORD *)(v12 + 32);
        if ( !v12 )
          return 1;
        if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 && (*(_BYTE *)(v12 + 4) & 8) == 0 )
        {
          v13 = *(_QWORD *)(v12 + 16);
          if ( v14 != v13 )
            goto LABEL_19;
        }
      }
    }
  }
  return 0;
}
