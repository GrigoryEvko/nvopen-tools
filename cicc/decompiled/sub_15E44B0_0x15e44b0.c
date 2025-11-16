// Function: sub_15E44B0
// Address: 0x15e44b0
//
__int64 __fastcall sub_15E44B0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // rax
  _BYTE *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  _QWORD *v12; // rcx

  v1 = sub_1626AA0(a1, 2);
  if ( !v1 )
    return -1;
  v2 = v1;
  v3 = -(__int64)*(unsigned int *)(v1 + 8);
  v4 = *(_BYTE **)(v2 + 8 * v3);
  if ( !v4 || *v4 )
    return -1;
  v5 = sub_161E970(*(_QWORD *)(v2 + 8 * v3));
  if ( v6 != 20
    || *(_QWORD *)v5 ^ 0x6E6F6974636E7566LL | *(_QWORD *)(v5 + 8) ^ 0x635F7972746E655FLL
    || *(_DWORD *)(v5 + 16) != 1953396079 )
  {
    v7 = sub_161E970(v4);
    if ( v8 == 30
      && !(*(_QWORD *)v7 ^ 0x69746568746E7973LL | *(_QWORD *)(v7 + 8) ^ 0x6974636E75665F63LL)
      && *(_QWORD *)(v7 + 16) == 0x7972746E655F6E6FLL
      && *(_DWORD *)(v7 + 24) == 1970234207
      && *(_WORD *)(v7 + 28) == 29806 )
    {
      v10 = *(_QWORD *)(*(_QWORD *)(v2 + 8 * (1LL - *(unsigned int *)(v2 + 8))) + 136LL);
      result = *(_QWORD *)(v10 + 24);
      if ( *(_DWORD *)(v10 + 32) > 0x40u )
        return *(_QWORD *)result;
      return result;
    }
    return -1;
  }
  v11 = *(_QWORD *)(*(_QWORD *)(v2 + 8 * (1LL - *(unsigned int *)(v2 + 8))) + 136LL);
  v12 = *(_QWORD **)(v11 + 24);
  if ( *(_DWORD *)(v11 + 32) > 0x40u )
    v12 = (_QWORD *)*v12;
  result = (__int64)v12;
  if ( v12 == (_QWORD *)-1LL )
    return -1;
  return result;
}
