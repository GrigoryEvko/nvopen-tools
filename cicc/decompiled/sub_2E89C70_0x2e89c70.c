// Function: sub_2E89C70
// Address: 0x2e89c70
//
__int64 __fastcall sub_2E89C70(__int64 a1, unsigned int a2, __int64 a3, char a4)
{
  int v5; // r14d
  unsigned int v8; // r8d
  __int64 v9; // rbx
  __int64 v10; // r12
  unsigned int v11; // esi
  char v13; // al
  unsigned int v14; // [rsp+0h] [rbp-40h]
  char v15; // [rsp+7h] [rbp-39h]

  v5 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  if ( !v5 )
    return 0xFFFFFFFFLL;
  v8 = a2 - 1;
  v9 = 0;
  while ( 1 )
  {
    v10 = *(_QWORD *)(a1 + 32) + 40 * v9;
    if ( !*(_BYTE *)v10 && (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
    {
      v11 = *(_DWORD *)(v10 + 8);
      if ( v11 )
      {
        if ( a2 == v11
          || a3
          && a2
          && v11 - 1 <= 0x3FFFFFFE
          && v8 <= 0x3FFFFFFE
          && (v14 = v8, v15 = a4, v13 = sub_E92070(a3, v11, a2), a4 = v15, v8 = v14, v13) )
        {
          if ( !a4 || (((*(_BYTE *)(v10 + 3) & 0x40) != 0) & ((*(_BYTE *)(v10 + 3) >> 4) ^ 1)) != 0 )
            break;
        }
      }
    }
    if ( v5 == (_DWORD)++v9 )
      return 0xFFFFFFFFLL;
  }
  return (unsigned int)v9;
}
