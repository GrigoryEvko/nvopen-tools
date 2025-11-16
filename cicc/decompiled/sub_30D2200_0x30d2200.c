// Function: sub_30D2200
// Address: 0x30d2200
//
_BYTE *__fastcall sub_30D2200(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // r8
  int v4; // ecx
  unsigned int v5; // edx
  __int64 *v6; // rax
  __int64 v7; // rdi
  _BYTE *result; // rax
  int v9; // eax
  int v10; // r9d

  v2 = *(_DWORD *)(a1 + 160);
  v3 = *(_QWORD *)(a1 + 144);
  if ( !v2 )
    return 0;
  v4 = v2 - 1;
  v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (__int64 *)(v3 + 16LL * v5);
  v7 = *v6;
  if ( a2 != *v6 )
  {
    v9 = 1;
    while ( v7 != -4096 )
    {
      v10 = v9 + 1;
      v5 = v4 & (v9 + v5);
      v6 = (__int64 *)(v3 + 16LL * v5);
      v7 = *v6;
      if ( a2 == *v6 )
        goto LABEL_3;
      v9 = v10;
    }
    return 0;
  }
LABEL_3:
  result = (_BYTE *)v6[1];
  if ( result )
  {
    if ( *result != 17 )
      return 0;
  }
  return result;
}
