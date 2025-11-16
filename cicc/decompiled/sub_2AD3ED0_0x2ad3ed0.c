// Function: sub_2AD3ED0
// Address: 0x2ad3ed0
//
__int64 *__fastcall sub_2AD3ED0(__int64 a1, __int64 *a2)
{
  bool v3; // zf
  __int64 *v4; // rax
  __int64 *result; // rax
  int v6; // ecx
  unsigned int v7; // esi
  int v8; // edx
  __int64 v9; // rdx
  __int64 *v10; // [rsp+0h] [rbp-20h] BYREF
  __int64 *v11; // [rsp+8h] [rbp-18h] BYREF

  v3 = (unsigned __int8)sub_2AC19D0(a1, a2, &v10) == 0;
  v4 = v10;
  if ( !v3 )
    return v10 + 2;
  v6 = *(_DWORD *)(a1 + 16);
  v7 = *(_DWORD *)(a1 + 24);
  v11 = v10;
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
  sub_2AD3C30(a1, v7);
  sub_2AC19D0(a1, a2, &v11);
  v8 = *(_DWORD *)(a1 + 16) + 1;
  v4 = v11;
LABEL_5:
  *(_DWORD *)(a1 + 16) = v8;
  if ( *v4 != -4096 || v4[1] != -4096 )
    --*(_DWORD *)(a1 + 20);
  result = v4 + 2;
  *(result - 2) = *a2;
  v9 = a2[1];
  *result = 0;
  *(result - 1) = v9;
  return result;
}
