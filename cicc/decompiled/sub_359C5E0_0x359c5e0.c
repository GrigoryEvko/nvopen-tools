// Function: sub_359C5E0
// Address: 0x359c5e0
//
__int64 *__fastcall sub_359C5E0(__int64 a1, __int64 *a2)
{
  unsigned int v3; // esi
  __int64 v4; // r8
  __int64 *v5; // r10
  int v6; // r11d
  unsigned int v7; // ecx
  __int64 *v8; // rax
  __int64 v9; // rdx
  int v11; // eax
  int v12; // edx
  __int64 v13; // rax
  __int64 *v14; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
    v14 = 0;
LABEL_18:
    v3 *= 2;
    goto LABEL_19;
  }
  v4 = *(_QWORD *)(a1 + 8);
  v5 = 0;
  v6 = 1;
  v7 = (v3 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v8 = (__int64 *)(v4 + 16LL * v7);
  v9 = *v8;
  if ( *a2 == *v8 )
    return v8 + 1;
  while ( v9 != -4096 )
  {
    if ( v9 == -8192 && !v5 )
      v5 = v8;
    v7 = (v3 - 1) & (v6 + v7);
    v8 = (__int64 *)(v4 + 16LL * v7);
    v9 = *v8;
    if ( *a2 == *v8 )
      return v8 + 1;
    ++v6;
  }
  if ( !v5 )
    v5 = v8;
  v11 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  v14 = v5;
  if ( 4 * (v11 + 1) >= 3 * v3 )
    goto LABEL_18;
  if ( v3 - *(_DWORD *)(a1 + 20) - v12 <= v3 >> 3 )
  {
LABEL_19:
    sub_2E48800(a1, v3);
    sub_3547B30(a1, a2, &v14);
    v5 = v14;
    v12 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v12;
  if ( *v5 != -4096 )
    --*(_DWORD *)(a1 + 20);
  v13 = *a2;
  v5[1] = 0;
  *v5 = v13;
  return v5 + 1;
}
