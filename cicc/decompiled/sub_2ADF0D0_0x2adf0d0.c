// Function: sub_2ADF0D0
// Address: 0x2adf0d0
//
__int64 __fastcall sub_2ADF0D0(__int64 a1, __int64 *a2)
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
  __int64 *v13; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    v13 = 0;
LABEL_17:
    v4 *= 2;
    goto LABEL_18;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 0;
  v7 = 1;
  result = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v9 = (__int64 *)(v5 + 8 * result);
  v10 = *v9;
  if ( *a2 == *v9 )
    return result;
  while ( v10 != -4096 )
  {
    if ( v6 || v10 != -8192 )
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
  v13 = v6;
  if ( 4 * (v11 + 1) >= 3 * v4 )
    goto LABEL_17;
  if ( v4 - *(_DWORD *)(a1 + 20) - v12 <= v4 >> 3 )
  {
LABEL_18:
    sub_CF4090(a1, v4);
    sub_23FDF60(a1, a2, &v13);
    v6 = v13;
    v12 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v12;
  if ( *v6 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v6 = *a2;
  return sub_9C95B0(a1 + 32, *a2);
}
