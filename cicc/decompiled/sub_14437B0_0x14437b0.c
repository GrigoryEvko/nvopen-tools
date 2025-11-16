// Function: sub_14437B0
// Address: 0x14437b0
//
__int64 __fastcall sub_14437B0(_QWORD *a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // r8
  int i; // eax
  int v13; // r9d

  v1 = *(_QWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 8);
  if ( !v1 )
    return 0;
  while ( 1 )
  {
    v2 = sub_1648700(v1);
    if ( (unsigned __int8)(*(_BYTE *)(v2 + 16) - 25) <= 9u )
      break;
    v1 = *(_QWORD *)(v1 + 8);
    if ( !v1 )
      return 0;
  }
  v3 = 0;
LABEL_5:
  v4 = a1[3];
  v5 = *(unsigned int *)(v4 + 48);
  if ( !(_DWORD)v5 )
    goto LABEL_12;
  v6 = *(_QWORD *)(v2 + 40);
  v7 = *(_QWORD *)(v4 + 32);
  v8 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v9 = (__int64 *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( v6 != *v9 )
  {
    for ( i = 1; ; i = v13 )
    {
      if ( v10 == -8 )
        goto LABEL_12;
      v13 = i + 1;
      v8 = (v5 - 1) & (i + v8);
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( v6 == *v9 )
        break;
    }
  }
  if ( v9 != (__int64 *)(v7 + 16 * v5) && v9[1] && !(unsigned __int8)sub_1443560(a1, v6) )
  {
    if ( !v3 )
    {
      v3 = v6;
      goto LABEL_12;
    }
    return 0;
  }
LABEL_12:
  while ( 1 )
  {
    v1 = *(_QWORD *)(v1 + 8);
    if ( !v1 )
      return v3;
    v2 = sub_1648700(v1);
    if ( (unsigned __int8)(*(_BYTE *)(v2 + 16) - 25) <= 9u )
      goto LABEL_5;
  }
}
