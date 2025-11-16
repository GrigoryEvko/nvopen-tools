// Function: sub_35A4950
// Address: 0x35a4950
//
__int64 *__fastcall sub_35A4950(__int64 a1, __int64 *a2)
{
  unsigned int v4; // r8d
  __int64 v5; // r9
  __int64 v6; // rdi
  __int64 *v7; // rcx
  int v8; // r11d
  unsigned int v9; // esi
  __int64 *v10; // rax
  __int64 v11; // rdx
  int v13; // eax
  int v14; // edx
  __int64 v15; // rax
  int v16; // esi
  __int64 *v17; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    v17 = 0;
LABEL_18:
    v16 = 2 * v4;
LABEL_19:
    sub_35A4050(a1, v16);
    sub_359BED0(a1, a2, &v17);
    v7 = v17;
    v14 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_14;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *a2;
  v7 = 0;
  v8 = 1;
  v9 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v10 = (__int64 *)(v5 + 80LL * v9);
  v11 = *v10;
  if ( v6 == *v10 )
    return v10 + 1;
  while ( v11 != -4096 )
  {
    if ( v11 == -8192 && !v7 )
      v7 = v10;
    v9 = (v4 - 1) & (v8 + v9);
    v10 = (__int64 *)(v5 + 80LL * v9);
    v11 = *v10;
    if ( v6 == *v10 )
      return v10 + 1;
    ++v8;
  }
  if ( !v7 )
    v7 = v10;
  v13 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v14 = v13 + 1;
  v17 = v7;
  if ( 4 * (v13 + 1) >= 3 * v4 )
    goto LABEL_18;
  if ( v4 - *(_DWORD *)(a1 + 20) - v14 <= v4 >> 3 )
  {
    v16 = v4;
    goto LABEL_19;
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v14;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a1 + 20);
  v15 = *a2;
  v7[9] = 0;
  *(_OWORD *)(v7 + 3) = 0;
  *v7 = v15;
  v7[1] = (__int64)(v7 + 3);
  v7[2] = 0x600000000LL;
  *(_OWORD *)(v7 + 5) = 0;
  *(_OWORD *)(v7 + 7) = 0;
  return v7 + 1;
}
