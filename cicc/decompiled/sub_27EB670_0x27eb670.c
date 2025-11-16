// Function: sub_27EB670
// Address: 0x27eb670
//
__int64 __fastcall sub_27EB670(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r9
  __int64 v10; // r13
  __int64 v11; // rbx
  int v13; // edx
  int v14; // r10d

  v4 = *(unsigned int *)(a2 + 120);
  v5 = *(_QWORD *)(a2 + 104);
  if ( !(_DWORD)v4 )
    return 0;
  v7 = (v4 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( a1 != *v8 )
  {
    v13 = 1;
    while ( v9 != -4096 )
    {
      v14 = v13 + 1;
      v7 = (v4 - 1) & (v13 + v7);
      v8 = (__int64 *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( a1 == *v8 )
        goto LABEL_3;
      v13 = v14;
    }
    return 0;
  }
LABEL_3:
  if ( v8 == (__int64 *)(v5 + 16 * v4) )
    return 0;
  v10 = v8[1];
  if ( !v10 )
    return 0;
  v11 = *(_QWORD *)(v10 + 8);
  if ( v11 == v10 )
    return 0;
  while ( 1 )
  {
    if ( !v11 )
      BUG();
    if ( *(_BYTE *)(v11 - 48) == 27 && (*(_QWORD *)(v11 + 16) != *(_QWORD *)(a3 + 64) || !sub_1041270(a2, v11 - 48, a3)) )
      break;
    v11 = *(_QWORD *)(v11 + 8);
    if ( v10 == v11 )
      return 0;
  }
  return 1;
}
