// Function: sub_227C140
// Address: 0x227c140
//
__int64 __fastcall sub_227C140(__int64 a1)
{
  int v1; // eax
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 i; // rdx
  unsigned int v5; // ecx
  unsigned int v6; // eax
  int v7; // eax
  unsigned __int64 v8; // r13

  v1 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v1 )
  {
    v5 = 4 * v1;
    v3 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)(4 * v1) < 0x40 )
      v5 = 64;
    if ( v5 >= (unsigned int)v3 )
    {
LABEL_4:
      result = *(_QWORD *)(a1 + 8);
      for ( i = result + 24 * v3; result != i; *(_QWORD *)(result - 16) = -4096 )
      {
        *(_QWORD *)result = -4096;
        result += 24;
      }
      goto LABEL_6;
    }
    v6 = v1 - 1;
    if ( v6 )
    {
      _BitScanReverse(&v6, v6);
      v7 = 1 << (33 - (v6 ^ 0x1F));
      if ( v7 < 64 )
        v7 = 64;
      if ( v7 == (_DWORD)v3 )
        return (__int64)sub_227C100(a1);
      v8 = 4 * v7 / 3u + 1;
    }
    else
    {
      v8 = 86;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 8), 24 * v3, 8);
    result = sub_AF1560(v8);
    *(_DWORD *)(a1 + 24) = result;
    if ( !(_DWORD)result )
      goto LABEL_20;
    *(_QWORD *)(a1 + 8) = sub_C7D670(24LL * (unsigned int)result, 8);
    return (__int64)sub_227C100(a1);
  }
  result = *(unsigned int *)(a1 + 20);
  if ( !(_DWORD)result )
    return result;
  v3 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)v3 <= 0x40 )
    goto LABEL_4;
  result = sub_C7D6A0(*(_QWORD *)(a1 + 8), 24 * v3, 8);
  *(_DWORD *)(a1 + 24) = 0;
LABEL_20:
  *(_QWORD *)(a1 + 8) = 0;
LABEL_6:
  *(_QWORD *)(a1 + 16) = 0;
  return result;
}
