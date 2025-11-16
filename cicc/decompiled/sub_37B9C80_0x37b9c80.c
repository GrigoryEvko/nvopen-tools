// Function: sub_37B9C80
// Address: 0x37b9c80
//
__int16 __fastcall sub_37B9C80(__int64 a1, unsigned int a2, unsigned __int8 a3)
{
  __int16 result; // ax
  __int64 v5; // rax
  unsigned int v7; // esi
  char *v8; // rax
  __int64 v9; // rdx
  char *v10; // rdi

  if ( a2 == -1 )
    return 0;
  if ( a3 > 2u )
    return 0;
  v5 = *(_QWORD *)(a1 + 16);
  v7 = *(_DWORD *)(*(_QWORD *)(v5 + 88) + 4LL * a2);
  if ( v7 >= *(_DWORD *)(v5 + 284) )
    return 259;
  if ( a3 == 2 )
    return 0;
  v8 = sub_E922F0(*(_QWORD **)(a1 + 3616), v7);
  v10 = &v8[2 * v9];
  if ( v8 == v10 )
  {
LABEL_13:
    LOBYTE(result) = 1;
    HIBYTE(result) = a3 ^ 1;
  }
  else
  {
    while ( (*(_QWORD *)(**(_QWORD **)(a1 + 3624) + 8 * ((unsigned __int64)*(unsigned __int16 *)v8 >> 6))
           & (1LL << *(_WORD *)v8)) == 0 )
    {
      v8 += 2;
      if ( v10 == v8 )
        goto LABEL_13;
    }
    return 258;
  }
  return result;
}
