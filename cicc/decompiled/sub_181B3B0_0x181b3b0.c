// Function: sub_181B3B0
// Address: 0x181b3b0
//
__int64 *__fastcall sub_181B3B0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r15
  unsigned int v4; // eax
  _BYTE *v5; // r12
  unsigned int v6; // esi
  __int64 v7; // rdi
  unsigned int v8; // edx
  __int64 *result; // rax
  __int64 v10; // rcx
  __int64 *v11; // rax
  int v12; // r14d
  unsigned int i; // r13d
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  int v17; // r11d
  __int64 *v18; // r10
  int v19; // ecx
  int v20; // ecx
  __int64 v21; // [rsp+10h] [rbp-40h] BYREF
  __int64 *v22; // [rsp+18h] [rbp-38h] BYREF

  v2 = a2;
  v3 = *a1;
  v4 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( v4 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v11 = *(__int64 **)(a2 - 8);
    else
      v11 = (__int64 *)(a2 - 24LL * v4);
    v5 = (_BYTE *)sub_1819D40((_QWORD *)v3, *v11);
    v12 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( v12 != 1 )
    {
      for ( i = 1; i != v12; ++i )
      {
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v14 = *(_QWORD *)(a2 - 8);
        else
          v14 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v15 = i;
        v16 = sub_1819D40((_QWORD *)v3, *(_QWORD *)(v14 + 24 * v15));
        v5 = (_BYTE *)sub_181A560((__int64 *)v3, v5, v16, a2);
      }
    }
    v3 = *a1;
  }
  else
  {
    v5 = *(_BYTE **)(*(_QWORD *)v3 + 200LL);
  }
  v6 = *(_DWORD *)(v3 + 152);
  v21 = v2;
  if ( !v6 )
  {
    ++*(_QWORD *)(v3 + 128);
LABEL_26:
    v6 *= 2;
    goto LABEL_27;
  }
  v7 = *(_QWORD *)(v3 + 136);
  v8 = (v6 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  result = (__int64 *)(v7 + 16LL * v8);
  v10 = *result;
  if ( v2 == *result )
    goto LABEL_5;
  v17 = 1;
  v18 = 0;
  while ( v10 != -8 )
  {
    if ( !v18 && v10 == -16 )
      v18 = result;
    v8 = (v6 - 1) & (v17 + v8);
    result = (__int64 *)(v7 + 16LL * v8);
    v10 = *result;
    if ( v2 == *result )
      goto LABEL_5;
    ++v17;
  }
  v19 = *(_DWORD *)(v3 + 144);
  if ( v18 )
    result = v18;
  ++*(_QWORD *)(v3 + 128);
  v20 = v19 + 1;
  if ( 4 * v20 >= 3 * v6 )
    goto LABEL_26;
  if ( v6 - *(_DWORD *)(v3 + 148) - v20 <= v6 >> 3 )
  {
LABEL_27:
    sub_176F940(v3 + 128, v6);
    sub_176A9A0(v3 + 128, &v21, &v22);
    result = v22;
    v2 = v21;
    v20 = *(_DWORD *)(v3 + 144) + 1;
  }
  *(_DWORD *)(v3 + 144) = v20;
  if ( *result != -8 )
    --*(_DWORD *)(v3 + 148);
  *result = v2;
  result[1] = 0;
LABEL_5:
  result[1] = (__int64)v5;
  return result;
}
