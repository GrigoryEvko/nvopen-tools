// Function: sub_2D2E9F0
// Address: 0x2d2e9f0
//
__int64 *__fastcall sub_2D2E9F0(__int64 a1, __int64 *a2)
{
  bool v3; // zf
  __int64 *v4; // rax
  __int64 *result; // rax
  int v6; // ecx
  unsigned int v7; // esi
  int v8; // edx
  __int64 v9; // rdx
  __int64 *v10; // rdx
  __int64 *v11; // [rsp+0h] [rbp-20h] BYREF
  __int64 *v12; // [rsp+8h] [rbp-18h] BYREF

  v3 = (unsigned __int8)sub_2D28EE0(a1, a2, &v11) == 0;
  v4 = v11;
  if ( !v3 )
    return v11 + 2;
  v6 = *(_DWORD *)(a1 + 16);
  v7 = *(_DWORD *)(a1 + 24);
  v12 = v11;
  ++*(_QWORD *)a1;
  v8 = v6 + 1;
  if ( 4 * (v6 + 1) >= 3 * v7 )
  {
    v7 *= 2;
  }
  else if ( v7 - *(_DWORD *)(a1 + 20) - v8 > v7 >> 3 )
  {
    goto LABEL_5;
  }
  sub_2D2E6F0(a1, v7);
  sub_2D28EE0(a1, a2, &v12);
  v8 = *(_DWORD *)(a1 + 16) + 1;
  v4 = v12;
LABEL_5:
  *(_DWORD *)(a1 + 16) = v8;
  if ( *v4 != -4096 || v4[1] != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v4 = *a2;
  v9 = a2[1];
  v4[3] = 0x800000000LL;
  v4[1] = v9;
  v10 = v4 + 4;
  result = v4 + 2;
  *result = (__int64)v10;
  return result;
}
