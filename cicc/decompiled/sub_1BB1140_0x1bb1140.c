// Function: sub_1BB1140
// Address: 0x1bb1140
//
__int64 __fastcall sub_1BB1140(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  __int64 *v6; // r10
  int v7; // r11d
  __int64 result; // rax
  __int64 *v9; // rdi
  __int64 v10; // rcx
  int v11; // eax
  int v12; // edx
  _BYTE *v13; // rsi
  __int64 *v14; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
LABEL_20:
    v4 *= 2;
    goto LABEL_21;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 0;
  v7 = 1;
  result = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v9 = (__int64 *)(v5 + 8 * result);
  v10 = *v9;
  if ( *a2 == *v9 )
    return result;
  while ( v10 != -8 )
  {
    if ( v10 != -16 || v6 )
      v9 = v6;
    result = (v4 - 1) & (v7 + (_DWORD)result);
    v10 = *(_QWORD *)(v5 + 8LL * (unsigned int)result);
    if ( *a2 == v10 )
      return result;
    ++v7;
    v6 = v9;
    v9 = (__int64 *)(v5 + 8LL * (unsigned int)result);
  }
  v11 = *(_DWORD *)(a1 + 16);
  if ( !v6 )
    v6 = v9;
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  if ( 4 * (v11 + 1) >= 3 * v4 )
    goto LABEL_20;
  if ( v4 - *(_DWORD *)(a1 + 20) - v12 <= v4 >> 3 )
  {
LABEL_21:
    sub_1467110(a1, v4);
    sub_1463A20(a1, a2, &v14);
    v6 = v14;
    v12 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v12;
  if ( *v6 != -8 )
    --*(_DWORD *)(a1 + 20);
  result = *a2;
  *v6 = *a2;
  v13 = *(_BYTE **)(a1 + 40);
  if ( v13 == *(_BYTE **)(a1 + 48) )
    return (__int64)sub_170B610(a1 + 32, v13, a2);
  if ( v13 )
  {
    *(_QWORD *)v13 = result;
    v13 = *(_BYTE **)(a1 + 40);
  }
  *(_QWORD *)(a1 + 40) = v13 + 8;
  return result;
}
