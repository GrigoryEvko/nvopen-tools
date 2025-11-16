// Function: sub_1E6D280
// Address: 0x1e6d280
//
__int64 __fastcall sub_1E6D280(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r14
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // r10
  __int64 v14; // rax
  unsigned int v15; // ecx
  char v16; // al
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rbx
  __int64 v20; // r13

  v7 = *a1;
  v9 = *(_QWORD *)(*a1 + 8);
  if ( v9 == *(_QWORD *)(*a1 + 16) )
    goto LABEL_7;
  v10 = *(unsigned int *)(v9 + 8LL * *(unsigned int *)(a2 + 192) + 4);
  v11 = *(unsigned int *)(v9 + 8LL * *(unsigned int *)(a3 + 192) + 4);
  if ( (_DWORD)v11 == (_DWORD)v10 )
    goto LABEL_7;
  a5 = (unsigned int)v11 >> 6;
  v12 = *(_QWORD *)a1[1];
  LOBYTE(a5) = (*(_QWORD *)(v12 + 8 * a5) & (1LL << v11)) != 0;
  if ( (_BYTE)a5 != ((*(_QWORD *)(v12 + 8LL * ((unsigned int)v10 >> 6)) & (1LL << v10)) != 0) )
    return (unsigned int)a5;
  v14 = *(_QWORD *)(v7 + 200);
  v15 = *(_DWORD *)(v14 + 4 * v10);
  if ( *(_DWORD *)(v14 + 4 * v11) == v15 )
  {
LABEL_7:
    v16 = *(_BYTE *)(a3 + 236) & 1;
    if ( *((_BYTE *)a1 + 16) )
    {
      if ( !v16 )
      {
        sub_1F01DD0(a3);
        v9 = *(_QWORD *)(v7 + 8);
        v7 = *a1;
      }
      v17 = (unsigned int)(*(_DWORD *)(a3 + 240) + 1);
      v18 = *(unsigned int *)(v9 + 8LL * *(unsigned int *)(a3 + 192));
      if ( (*(_BYTE *)(a2 + 236) & 1) == 0 )
        sub_1F01DD0(a2);
      LOBYTE(a5) = (unsigned __int64)*(unsigned int *)(*(_QWORD *)(v7 + 8) + 8LL * *(unsigned int *)(a2 + 192)) * v17 < v18 * (unsigned __int64)(unsigned int)(*(_DWORD *)(a2 + 240) + 1);
      return (unsigned int)a5;
    }
    else
    {
      if ( !v16 )
      {
        sub_1F01DD0(a3);
        v9 = *(_QWORD *)(v7 + 8);
        v7 = *a1;
      }
      v19 = (unsigned int)(*(_DWORD *)(a3 + 240) + 1);
      v20 = *(unsigned int *)(v9 + 8LL * *(unsigned int *)(a3 + 192));
      if ( (*(_BYTE *)(a2 + 236) & 1) == 0 )
        sub_1F01DD0(a2);
      LOBYTE(a5) = v20 * (unsigned __int64)(unsigned int)(*(_DWORD *)(a2 + 240) + 1) < (unsigned __int64)*(unsigned int *)(*(_QWORD *)(v7 + 8) + 8LL * *(unsigned int *)(a2 + 192))
                                                                                     * v19;
      return (unsigned int)a5;
    }
  }
  else
  {
    LOBYTE(a5) = *(_DWORD *)(v14 + 4 * v11) > v15;
    return (unsigned int)a5;
  }
}
