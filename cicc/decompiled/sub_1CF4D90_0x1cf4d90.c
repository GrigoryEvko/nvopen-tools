// Function: sub_1CF4D90
// Address: 0x1cf4d90
//
__int64 __fastcall sub_1CF4D90(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // r15
  bool v8; // al
  unsigned int v9; // ecx
  unsigned int v10; // r8d
  unsigned int v11; // r9d
  __int64 v12; // rbx
  __int64 v13; // r15
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rsi
  bool v17; // al
  int v18; // eax
  unsigned int v19; // [rsp+0h] [rbp-40h]
  int v20; // [rsp+4h] [rbp-3Ch]
  unsigned int v21; // [rsp+8h] [rbp-38h]
  unsigned int v22; // [rsp+Ch] [rbp-34h]

  v5 = *a1;
  if ( !*a1 )
    return sub_134CB50(a3, (__int64)a1, (__int64)a2);
  if ( *(_BYTE *)(v5 + 16) != 56 )
    return sub_134CB50(a3, (__int64)a1, (__int64)a2);
  v6 = *a2;
  if ( !*a2 )
    return sub_134CB50(a3, (__int64)a1, (__int64)a2);
  if ( *(_BYTE *)(v6 + 16) != 56 )
    return sub_134CB50(a3, (__int64)a1, (__int64)a2);
  if ( !sub_15FA300(*a1) )
    return sub_134CB50(a3, (__int64)a1, (__int64)a2);
  v8 = sub_15FA300(v6);
  if ( v5 == v6 )
    return sub_134CB50(a3, (__int64)a1, (__int64)a2);
  if ( !v8 )
    return sub_134CB50(a3, (__int64)a1, (__int64)a2);
  v9 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
  v10 = *(_DWORD *)(v6 + 20) & 0xFFFFFFF;
  if ( *(_QWORD *)(v5 - 24LL * v9) != *(_QWORD *)(v6 - 24LL * v10) )
    return sub_134CB50(a3, (__int64)a1, (__int64)a2);
  v11 = *(_DWORD *)(v6 + 20) & 0xFFFFFFF;
  if ( v9 <= v10 )
    v11 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
  if ( v11 > 1 )
  {
    v12 = v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF) + 24;
    v13 = v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF) + 24;
    v14 = 1;
    while ( 1 )
    {
      v15 = *(_QWORD *)v12;
      if ( *(_BYTE *)(*(_QWORD *)v12 + 16LL) != 13 )
        break;
      v16 = *(_QWORD *)v13;
      if ( *(_BYTE *)(*(_QWORD *)v13 + 16LL) != 13 )
        break;
      if ( *(_DWORD *)(v15 + 32) <= 0x40u )
      {
        if ( *(_QWORD *)(v15 + 24) != *(_QWORD *)(v16 + 24) )
          return 0;
      }
      else
      {
        v19 = v11;
        v20 = v14;
        v21 = v10;
        v22 = v9;
        v17 = sub_16A5220(v15 + 24, (const void **)(v16 + 24));
        v9 = v22;
        v10 = v21;
        v14 = v20;
        v11 = v19;
        if ( !v17 )
          return 0;
      }
      ++v14;
      v12 += 24;
      v13 += 24;
      if ( v14 == v11 )
      {
        v18 = v11;
        goto LABEL_20;
      }
    }
    return sub_134CB50(a3, (__int64)a1, (__int64)a2);
  }
  v18 = 1;
LABEL_20:
  if ( v9 == v10 || v18 != v11 )
    return sub_134CB50(a3, (__int64)a1, (__int64)a2);
  return 3;
}
