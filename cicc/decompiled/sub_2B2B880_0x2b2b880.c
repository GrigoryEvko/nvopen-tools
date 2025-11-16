// Function: sub_2B2B880
// Address: 0x2b2b880
//
__int64 __fastcall sub_2B2B880(__int64 a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 *v6; // rax
  __int64 *v7; // rcx
  int v8; // esi
  __int64 *i; // rax
  __int64 *v10; // rdx
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // [rsp+0h] [rbp-40h] BYREF
  __int64 v20; // [rsp+8h] [rbp-38h]

  v2 = 1;
  v4 = a2;
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 8) - 15) <= 2u )
  {
    while ( !(unsigned __int8)sub_BCADB0(v4) )
    {
      if ( *(_BYTE *)(v4 + 8) == 15 )
      {
        v5 = *(unsigned int *)(v4 + 12);
        v6 = *(__int64 **)(v4 + 16);
        v7 = &v6[v5];
        v4 = *v6;
        v8 = v5;
        if ( v6 != v7 )
        {
          for ( i = v6 + 1; ; ++i )
          {
            v10 = i;
            if ( i == v7 )
              break;
            if ( v4 != *v10 )
              return 0;
          }
        }
        v2 *= v8;
      }
      else
      {
        v2 *= *(_DWORD *)(v4 + 32);
        v4 = *(_QWORD *)(v4 + 24);
      }
      if ( (unsigned __int8)(*(_BYTE *)(v4 + 8) - 15) > 2u )
        goto LABEL_12;
    }
    return 0;
  }
LABEL_12:
  if ( !(unsigned __int8)sub_BCBCB0(v4) )
    return 0;
  if ( (*(_BYTE *)(v4 + 8) & 0xFD) == 4 )
    return 0;
  v12 = *(_QWORD *)(a1 + 3344);
  v13 = sub_2B08680(v4, v2);
  v14 = sub_9208B0(v12, v13);
  v20 = v15;
  v19 = (v14 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v16 = sub_CA1930(&v19);
  if ( *(unsigned int *)(a1 + 3364) > v16 )
    return 0;
  if ( *(unsigned int *)(a1 + 3360) < v16 )
    return 0;
  v17 = sub_9208B0(*(_QWORD *)(a1 + 3344), a2);
  v20 = v18;
  v19 = (v17 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( sub_CA1930(&v19) != v16 )
    return 0;
  return v2;
}
