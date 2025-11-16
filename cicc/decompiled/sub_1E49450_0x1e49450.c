// Function: sub_1E49450
// Address: 0x1e49450
//
__int64 __fastcall sub_1E49450(__int64 a1, unsigned int a2, unsigned int a3, int a4, int a5, __int64 a6, __int64 a7)
{
  int v11; // eax
  int v12; // r8d
  int v13; // r9d
  int v14; // ecx
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-68h]
  __int64 v18; // [rsp+20h] [rbp-50h]
  int v19[3]; // [rsp+2Ch] [rbp-44h] BYREF
  _QWORD v20[7]; // [rsp+38h] [rbp-38h] BYREF

  v19[0] = a4;
  if ( a2 <= a3 )
    return 0;
  v18 = sub_1E69D00(*(_QWORD *)(a1 + 40), (unsigned int)v19[0]);
  if ( a3 == a5 )
  {
    v17 = a6 + 32LL * (a2 - 1);
    if ( (unsigned __int8)sub_1932870(v17, v19, v20) )
      return (unsigned int)sub_1E49390(v17, v19)[1];
  }
  if ( (unsigned __int8)sub_1932870(a6 + 32LL * a2, v19, v20) )
    return (unsigned int)sub_1E49390(a6 + 32LL * a2, v19)[1];
  if ( **(_WORD **)(v18 + 16) != 45 && **(_WORD **)(v18 + 16) || a7 != *(_QWORD *)(v18 + 24) )
    return (unsigned int)v19[0];
  if ( a3 + 1 != a2 )
  {
    if ( a3 + 1 < a2 )
    {
      v11 = sub_1E40FE0(*(_QWORD *)(v18 + 32), *(_DWORD *)(v18 + 40), a7);
      return sub_1E49450(a1, a2 - 1, a3, v11, v12, v13, a7);
    }
    return 0;
  }
  v14 = *(_DWORD *)(v18 + 40);
  v15 = *(_QWORD *)(v18 + 32);
  if ( v14 == 1 )
    return 0;
  v16 = 1;
  while ( a7 == *(_QWORD *)(v15 + 40LL * (unsigned int)(v16 + 1) + 24) )
  {
    v16 = (unsigned int)(v16 + 2);
    if ( v14 == (_DWORD)v16 )
      return 0;
  }
  return *(unsigned int *)(v15 + 40 * v16 + 8);
}
