// Function: sub_1BBD240
// Address: 0x1bbd240
//
__int64 __fastcall sub_1BBD240(__int64 *a1)
{
  __int64 v1; // rcx
  __int64 v2; // rax
  unsigned int v3; // r8d
  _QWORD *v4; // rdi
  _QWORD *v5; // rsi
  unsigned int v6; // r8d
  _QWORD *v7; // rax
  _QWORD *v8; // rax

  v1 = *a1;
  v2 = a1[1] - *a1;
  if ( v2 == 176 )
    return *(unsigned __int8 *)(v1 + 88) ^ 1u;
  v3 = 0;
  if ( v2 != 352 || *(_BYTE *)(v1 + 88) )
    return v3;
  v4 = *(_QWORD **)(v1 + 176);
  v5 = &v4[*(unsigned int *)(v1 + 184)];
  v6 = *(_DWORD *)(v1 + 184);
  if ( v5 != v4 )
  {
    v7 = *(_QWORD **)(v1 + 176);
    while ( *(_BYTE *)(*v7 + 16LL) <= 0x10u )
    {
      if ( v5 == ++v7 )
        return 1;
    }
    if ( v6 > 1 )
    {
      v8 = v4 + 1;
      while ( *v8 == *v4 )
      {
        if ( &v4[v6] == ++v8 )
          return 1;
      }
      return *(unsigned __int8 *)(v1 + 264) ^ 1u;
    }
  }
  return 1;
}
