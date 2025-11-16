// Function: sub_20C73B0
// Address: 0x20c73b0
//
__int64 __fastcall sub_20C73B0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned int v3; // edx
  __int64 v4; // rax
  unsigned int *v6; // rsi
  unsigned int v7; // ecx
  __int64 v8; // rax
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // r15
  int v13; // eax
  bool i; // al
  __int64 v15; // rax
  __int64 v16; // rax

  v3 = *(_DWORD *)(a2 + 8);
  if ( v3 )
  {
    v4 = *(unsigned int *)(a1 + 8);
    while ( 1 )
    {
      v6 = (unsigned int *)(*(_QWORD *)a2 + 4LL * v3 - 4);
      v7 = *v6 + 1;
      v8 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v4 - 8);
      LOBYTE(v2) = *(_BYTE *)(v8 + 8) == 14 ? *(_QWORD *)(v8 + 32) > (unsigned __int64)v7 : v7 < *(_DWORD *)(v8 + 12);
      if ( (_BYTE)v2 )
        break;
      *(_DWORD *)(a2 + 8) = v3 - 1;
      v4 = (unsigned int)(*(_DWORD *)(a1 + 8) - 1);
      *(_DWORD *)(a1 + 8) = v4;
      v3 = *(_DWORD *)(a2 + 8);
      if ( !v3 )
        return 0;
    }
    *v6 = v7;
    v12 = sub_1643D80(
            *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8),
            *(_DWORD *)(*(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8) - 4));
    v13 = *(unsigned __int8 *)(v12 + 8);
    if ( v13 == 13 )
      goto LABEL_19;
LABEL_11:
    if ( v13 == 14 )
    {
      for ( i = *(_QWORD *)(v12 + 32) != 0; i; i = *(_DWORD *)(v12 + 12) != 0 )
      {
        v15 = *(unsigned int *)(a1 + 8);
        if ( (unsigned int)v15 >= *(_DWORD *)(a1 + 12) )
        {
          sub_16CD150(a1, (const void *)(a1 + 16), 0, 8, v10, v11);
          v15 = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v15) = v12;
        ++*(_DWORD *)(a1 + 8);
        v16 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v16 >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, (const void *)(a2 + 16), 0, 4, v10, v11);
          v16 = *(unsigned int *)(a2 + 8);
        }
        *(_DWORD *)(*(_QWORD *)a2 + 4 * v16) = 0;
        ++*(_DWORD *)(a2 + 8);
        v12 = sub_1643D80(v12, 0);
        v13 = *(unsigned __int8 *)(v12 + 8);
        if ( v13 != 13 )
          goto LABEL_11;
LABEL_19:
        ;
      }
    }
  }
  else
  {
    return 0;
  }
  return v2;
}
