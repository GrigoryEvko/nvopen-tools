// Function: sub_31C7EB0
// Address: 0x31c7eb0
//
__int64 __fastcall sub_31C7EB0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v7; // r15
  bool v8; // al
  unsigned int v9; // ecx
  unsigned int v10; // edx
  _QWORD *v11; // rbx
  _QWORD *v12; // r15
  unsigned int v13; // r10d
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rsi
  bool v18; // al
  int v19; // eax
  unsigned int v20; // [rsp+4h] [rbp-4Ch]
  unsigned int v21; // [rsp+8h] [rbp-48h]
  unsigned int v22; // [rsp+Ch] [rbp-44h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+18h] [rbp-38h]

  v5 = *a1;
  if ( !*a1 )
    return sub_CF4E00(a3, (__int64)a1, (__int64)a2);
  if ( *(_BYTE *)v5 != 63 )
    return sub_CF4E00(a3, (__int64)a1, (__int64)a2);
  v7 = *a2;
  if ( !*a2 )
    return sub_CF4E00(a3, (__int64)a1, (__int64)a2);
  if ( *(_BYTE *)v7 != 63 )
    return sub_CF4E00(a3, (__int64)a1, (__int64)a2);
  if ( !sub_B4DE30(*a1) )
    return sub_CF4E00(a3, (__int64)a1, (__int64)a2);
  v8 = sub_B4DE30(v7);
  if ( v5 == v7 )
    return sub_CF4E00(a3, (__int64)a1, (__int64)a2);
  if ( !v8 )
    return sub_CF4E00(a3, (__int64)a1, (__int64)a2);
  v9 = *(_DWORD *)(v5 + 4) & 0x7FFFFFF;
  v10 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
  v11 = (_QWORD *)(v5 - 32LL * v9);
  v12 = (_QWORD *)(v7 - 32LL * v10);
  if ( *v11 != *v12 )
    return sub_CF4E00(a3, (__int64)a1, (__int64)a2);
  v13 = v10;
  if ( v9 <= v10 )
    v13 = v9;
  if ( v13 > 1 )
  {
    v14 = v13 + 1;
    v15 = 2;
    while ( 1 )
    {
      v16 = v11[4 * v15 - 4];
      if ( *(_BYTE *)v16 != 17 )
        break;
      v17 = v12[4 * v15 - 4];
      if ( *(_BYTE *)v17 != 17 )
        break;
      if ( *(_DWORD *)(v16 + 32) <= 0x40u )
      {
        if ( *(_QWORD *)(v16 + 24) != *(_QWORD *)(v17 + 24) )
          return 0;
      }
      else
      {
        v20 = v13;
        v21 = v10;
        v22 = v9;
        v23 = v15;
        v24 = v14;
        v18 = sub_C43C50(v16 + 24, (const void **)(v17 + 24));
        v14 = v24;
        v15 = v23;
        v9 = v22;
        v10 = v21;
        v13 = v20;
        if ( !v18 )
          return 0;
      }
      v19 = v15++;
      if ( v14 == v15 )
        goto LABEL_19;
    }
    return sub_CF4E00(a3, (__int64)a1, (__int64)a2);
  }
  v19 = 1;
LABEL_19:
  if ( v9 == v10 || v19 != v13 )
    return sub_CF4E00(a3, (__int64)a1, (__int64)a2);
  return 3;
}
