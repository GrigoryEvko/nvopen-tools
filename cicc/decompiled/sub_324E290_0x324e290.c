// Function: sub_324E290
// Address: 0x324e290
//
__int64 __fastcall sub_324E290(__int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  unsigned int v3; // ecx
  __int64 v4; // r8
  int v5; // [rsp-1Ch] [rbp-1Ch]

  result = a1[28];
  if ( !result )
  {
    v2 = sub_324C6D0(a1, 36, (__int64)(a1 + 1), 0);
    a1[28] = v2;
    sub_324AD70(a1, v2, 3, "__ARRAY_SIZE_TYPE__", 0x13u);
    BYTE2(v5) = 0;
    sub_3249A20(a1, (unsigned __int64 **)(a1[28] + 8), 11, v5, 8);
    v3 = *(unsigned __int16 *)(a1[10] + 16);
    if ( v3 > 0x2D )
      v4 = 7;
    else
      v4 = ((1LL << v3) & 0x200C00004180LL) == 0 ? 7LL : 5LL;
    sub_3249A20(a1, (unsigned __int64 **)(a1[28] + 8), 62, 65547, v4);
    sub_3238440(a1[26], (__int64)a1, *(_DWORD *)(a1[10] + 36), (__int64)"__ARRAY_SIZE_TYPE__", 19, a1[28]);
    return a1[28];
  }
  return result;
}
