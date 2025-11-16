// Function: sub_2169B90
// Address: 0x2169b90
//
unsigned __int64 __fastcall sub_2169B90(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r13
  unsigned int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // r15
  unsigned __int64 result; // rax
  char v13; // si
  unsigned int v14; // edi
  int v15; // r13d
  __int64 v16; // rcx
  int v17; // r14d
  int i; // r15d
  __int64 v19; // rdx
  __int64 v20; // [rsp+8h] [rbp-48h]
  int v23; // [rsp+18h] [rbp-38h]

  v8 = a1[2];
  v9 = sub_1F43D70(v8, a2);
  v11 = v9;
  if ( v9 == 134 && *(_BYTE *)(a4 + 8) == 16 )
    v11 = 135;
  result = sub_1F43D80(v8, *a1, a3, v10);
  v13 = *(_BYTE *)(a3 + 8);
  if ( v13 == 16 )
  {
    if ( (unsigned __int8)(BYTE4(result) - 14) > 0x5Fu )
      goto LABEL_12;
    v14 = BYTE4(result);
    if ( !*(_QWORD *)(v8 + 8LL * BYTE4(result) + 120) )
      goto LABEL_12;
  }
  else
  {
    if ( !BYTE4(result) )
      return 1;
    v14 = BYTE4(result);
    if ( !*(_QWORD *)(v8 + 8LL * BYTE4(result) + 120) )
      return 1;
  }
  if ( (unsigned int)v11 > 0x102 || *(_BYTE *)(v11 + v8 + 259LL * v14 + 2422) != 2 )
    return result;
  if ( v13 != 16 )
    return 1;
LABEL_12:
  v20 = *(_QWORD *)(a3 + 32);
  if ( a4 && *(_BYTE *)(a4 + 8) == 16 )
    a4 = **(_QWORD **)(a4 + 16);
  v15 = 0;
  v23 = sub_2169B90(a1, a2, **(_QWORD **)(a3 + 16), a4, a5);
  v17 = *(_QWORD *)(a3 + 32);
  if ( v17 > 0 )
  {
    for ( i = 0; i != v17; ++i )
    {
      v19 = a3;
      if ( *(_BYTE *)(a3 + 8) == 16 )
        v19 = **(_QWORD **)(a3 + 16);
      v15 += sub_1F43D80(a1[2], *a1, v19, v16);
    }
  }
  return (unsigned int)(v20 * v23 + v15);
}
