// Function: sub_2D26D70
// Address: 0x2d26d70
//
__int64 __fastcall sub_2D26D70(__int64 a1, __int64 a2)
{
  int v2; // r13d
  size_t v3; // rdx
  __int64 v5; // rcx
  _DWORD *v6; // rax
  _DWORD *v7; // rdx
  _DWORD *v8; // rcx
  __int64 *v9; // [rsp+10h] [rbp-50h] BYREF
  int v10; // [rsp+18h] [rbp-48h]
  __int64 v11; // [rsp+20h] [rbp-40h]
  int v12; // [rsp+28h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 64);
  if ( *(_DWORD *)(a2 + 64) != v2 )
    return 0;
  v3 = 8LL * *(unsigned int *)(a1 + 8);
  if ( v3 )
  {
    if ( memcmp(*(const void **)a1, *(const void **)a2, v3) )
      return 0;
  }
  v5 = *(unsigned int *)(a1 + 208);
  if ( v5 != *(_DWORD *)(a2 + 208) )
    return 0;
  v6 = *(_DWORD **)(a1 + 200);
  v7 = *(_DWORD **)(a2 + 200);
  v8 = &v6[v5];
  if ( v6 != v8 )
  {
    while ( *v6 == *v7 )
    {
      ++v6;
      ++v7;
      if ( v8 == v6 )
        goto LABEL_11;
    }
    return 0;
  }
LABEL_11:
  v9 = (__int64 *)a1;
  v10 = sub_2D26CA0((__int64 *)a1, 0, v2, 1);
  v11 = a1;
  v12 = -1;
  if ( !(unsigned __int8)sub_2D24140(&v9, (__int64 *)(a1 + 72), (__int64 *)(a2 + 72)) )
    return 0;
  v10 = sub_2D26CA0((__int64 *)a1, 0, v2, 1);
  return sub_2D24140(&v9, (__int64 *)(a1 + 136), (__int64 *)(a2 + 136));
}
