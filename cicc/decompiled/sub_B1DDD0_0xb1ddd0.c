// Function: sub_B1DDD0
// Address: 0xb1ddd0
//
__int64 *__fastcall sub_B1DDD0(__int64 a1, __int64 *a2)
{
  bool v3; // zf
  __int64 *v4; // rax
  __int64 *result; // rax
  unsigned int v6; // edx
  int v7; // edx
  unsigned int v8; // esi
  unsigned int v9; // ecx
  __int64 v10; // rdx
  __int64 *v11; // [rsp+0h] [rbp-20h] BYREF
  __int64 *v12; // [rsp+8h] [rbp-18h] BYREF

  v3 = (unsigned __int8)sub_B1C410(a1, a2, &v11) == 0;
  v4 = v11;
  if ( !v3 )
    return v11 + 2;
  v6 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v12 = v4;
  v7 = (v6 >> 1) + 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v9 = 12;
    v8 = 4;
  }
  else
  {
    v8 = *(_DWORD *)(a1 + 24);
    v9 = 3 * v8;
  }
  if ( 4 * v7 >= v9 )
  {
    v8 *= 2;
  }
  else if ( v8 - (v7 + *(_DWORD *)(a1 + 12)) > v8 >> 3 )
  {
    goto LABEL_7;
  }
  sub_B1DB20(a1, v8);
  sub_B1C410(a1, a2, &v12);
  v4 = v12;
  v7 = (*(_DWORD *)(a1 + 8) >> 1) + 1;
LABEL_7:
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v7);
  if ( *v4 != -4096 || v4[1] != -4096 )
    --*(_DWORD *)(a1 + 12);
  result = v4 + 2;
  *(result - 2) = *a2;
  v10 = a2[1];
  *(_DWORD *)result = 0;
  *(result - 1) = v10;
  return result;
}
