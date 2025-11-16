// Function: sub_19EC730
// Address: 0x19ec730
//
__int64 __fastcall sub_19EC730(__int64 a1, __int64 *a2)
{
  char v3; // r8
  __int64 result; // rax
  int v5; // ecx
  unsigned int v6; // esi
  int v7; // edx
  __int64 v8; // rdx
  _QWORD v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_154CC80(a1, a2, v9);
  result = v9[0];
  if ( v3 )
    return result;
  v5 = *(_DWORD *)(a1 + 16);
  v6 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)a1;
  v7 = v5 + 1;
  if ( 4 * (v5 + 1) >= 3 * v6 )
  {
    v6 *= 2;
    goto LABEL_8;
  }
  if ( v6 - *(_DWORD *)(a1 + 20) - v7 <= v6 >> 3 )
  {
LABEL_8:
    sub_1542080(a1, v6);
    sub_154CC80(a1, a2, v9);
    result = v9[0];
    v7 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v7;
  if ( *(_QWORD *)result != -8 )
    --*(_DWORD *)(a1 + 20);
  v8 = *a2;
  *(_DWORD *)(result + 8) = 0;
  *(_QWORD *)result = v8;
  return result;
}
