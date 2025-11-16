// Function: sub_8B30E0
// Address: 0x8b30e0
//
__int64 __fastcall sub_8B30E0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v7; // r14
  __int64 v8; // r15
  _QWORD *v9; // rbx
  int v10; // edx
  __int64 v11; // rax
  __int64 *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 *v15; // rdx
  __int64 v16; // r14
  __int64 v17; // r15

  if ( (*(_BYTE *)(a1 + 81) & 0x20) != 0 || (*(_BYTE *)(a2 + 81) & 0x20) != 0 )
    return 0;
  v7 = *(_QWORD *)(a1 + 88);
  v8 = *(_QWORD *)(a2 + 88);
  if ( (*(_BYTE *)(v7 + 160) & 2) != 0 || (*(_BYTE *)(v8 + 160) & 2) != 0 )
  {
    v16 = *(_QWORD *)(v7 + 104);
    v17 = *(_QWORD *)(v8 + 104);
    if ( !strcmp(*(const char **)(v16 + 8), *(const char **)(v17 + 8)) )
      return (unsigned int)sub_8B3500(
                             *(_QWORD *)(*(_QWORD *)(v16 + 40) + 32LL),
                             *(_QWORD *)(*(_QWORD *)(v17 + 40) + 32LL),
                             a3,
                             a4,
                             0) != 0;
    return 0;
  }
  if ( (*(_BYTE *)(v8 + 266) & 1) == 0 )
    return sub_89B9E0(v7, v8, 0, 0);
  v9 = sub_725090(2u);
  v9[4] = *(_QWORD *)(v7 + 104);
  sub_880AD0(a2)[8] = v8;
  if ( !(unsigned int)sub_8B2F00((__int64)v9, v8, *a3, a4, 1, 0) )
    return 0;
  v10 = 0;
  if ( a4 )
    v10 = *(_DWORD *)(sub_892BC0(a4) + 4);
  v11 = *(_QWORD *)(v8 + 104);
  if ( *(_DWORD *)(v11 + 132) != v10 )
    return 0;
  v12 = sub_8A4360(a4, a3, (unsigned int *)(v11 + 128), 0, 0);
  v13 = *(_QWORD *)(v7 + 104);
  v14 = v12[4];
  v15 = v12;
  result = v13 == v14;
  if ( !v14 )
  {
    v15[4] = v13;
    *((_BYTE *)v15 + 24) = (4 * *(_BYTE *)(v13 + 121)) & 0x10 | v15[3] & 0xEF;
    return 1;
  }
  return result;
}
