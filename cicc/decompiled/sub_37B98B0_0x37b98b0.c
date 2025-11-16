// Function: sub_37B98B0
// Address: 0x37b98b0
//
_QWORD *__fastcall sub_37B98B0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v4; // esi
  __int64 v5; // rcx
  __int64 v6; // rdi
  __int64 *v7; // r12
  int v8; // r10d
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  _QWORD *result; // rax
  int v13; // eax
  int v14; // edx
  __int64 v15; // rdx
  __int64 v16; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v17; // [rsp+18h] [rbp-28h] BYREF

  v2 = a1 + 64;
  v16 = a2;
  v4 = *(_DWORD *)(a1 + 88);
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 64);
    v17 = 0;
LABEL_20:
    v4 *= 2;
    goto LABEL_21;
  }
  v5 = v16;
  v6 = *(_QWORD *)(a1 + 72);
  v7 = 0;
  v8 = 1;
  v9 = (v4 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( v16 == *v10 )
    return (_QWORD *)v10[1];
  while ( v11 != -4096 )
  {
    if ( v11 == -8192 && !v7 )
      v7 = v10;
    v9 = (v4 - 1) & (v8 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v16 == *v10 )
      return (_QWORD *)v10[1];
    ++v8;
  }
  if ( !v7 )
    v7 = v10;
  v13 = *(_DWORD *)(a1 + 80);
  ++*(_QWORD *)(a1 + 64);
  v14 = v13 + 1;
  v17 = v7;
  if ( 4 * (v13 + 1) >= 3 * v4 )
    goto LABEL_20;
  if ( v4 - *(_DWORD *)(a1 + 84) - v14 <= v4 >> 3 )
  {
LABEL_21:
    sub_37B9700(v2, v4);
    sub_37B6040(v2, &v16, &v17);
    v5 = v16;
    v7 = v17;
    v14 = *(_DWORD *)(a1 + 80) + 1;
  }
  *(_DWORD *)(a1 + 80) = v14;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a1 + 84);
  *v7 = v5;
  v7[1] = 0;
  result = (_QWORD *)sub_22077B0(0x80u);
  if ( result )
  {
    v15 = v16;
    result[1] = a1;
    result[3] = 0x100000000LL;
    *result = v15;
    result[2] = result + 4;
  }
  v7[1] = (__int64)result;
  return result;
}
