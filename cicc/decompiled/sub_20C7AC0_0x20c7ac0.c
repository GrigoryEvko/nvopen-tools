// Function: sub_20C7AC0
// Address: 0x20c7ac0
//
__int64 __fastcall sub_20C7AC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r13
  int v8; // eax
  __int64 v9; // rax
  __int64 result; // rax
  __int64 v11; // rax
  __int64 v12; // rax

  v6 = a1;
  v8 = *(unsigned __int8 *)(a1 + 8);
  if ( v8 != 13 )
    goto LABEL_2;
  while ( *(_DWORD *)(v6 + 12) )
  {
    v11 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v11 < *(_DWORD *)(a2 + 12) )
      goto LABEL_11;
LABEL_16:
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, a5, a6);
    v11 = *(unsigned int *)(a2 + 8);
    while ( 1 )
    {
LABEL_11:
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v11) = v6;
      ++*(_DWORD *)(a2 + 8);
      v12 = *(unsigned int *)(a3 + 8);
      if ( (unsigned int)v12 >= *(_DWORD *)(a3 + 12) )
      {
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 4, a5, a6);
        v12 = *(unsigned int *)(a3 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a3 + 4 * v12) = 0;
      ++*(_DWORD *)(a3 + 8);
      v6 = sub_1643D80(v6, 0);
      v8 = *(unsigned __int8 *)(v6 + 8);
      if ( v8 == 13 )
        break;
LABEL_2:
      if ( v8 != 14 || !*(_QWORD *)(v6 + 32) )
        goto LABEL_3;
      v11 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v11 >= *(_DWORD *)(a2 + 12) )
        goto LABEL_16;
    }
  }
LABEL_3:
  v9 = *(unsigned int *)(a3 + 8);
  if ( (_DWORD)v9 )
  {
    while ( (unsigned int)*(unsigned __int8 *)(sub_1643D80(
                                                 *(_QWORD *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8) - 8),
                                                 *(_DWORD *)(*(_QWORD *)a3 + 4 * v9 - 4))
                                             + 8)
          - 13 <= 1 )
    {
      result = sub_20C73B0(a2, a3);
      if ( !(_BYTE)result )
        return result;
      v9 = *(unsigned int *)(a3 + 8);
    }
  }
  return 1;
}
