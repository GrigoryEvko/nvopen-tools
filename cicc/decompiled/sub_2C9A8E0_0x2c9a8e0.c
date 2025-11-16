// Function: sub_2C9A8E0
// Address: 0x2c9a8e0
//
__int64 *__fastcall sub_2C9A8E0(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // rcx
  int v6; // r11d
  __int64 v7; // rdi
  __int64 *v8; // r8
  unsigned int i; // eax
  __int64 *v10; // r9
  __int64 v11; // r13
  unsigned int v12; // eax
  int v14; // eax
  int v15; // edx
  __int64 v16; // rax
  __int64 *v17; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    v17 = 0;
    goto LABEL_23;
  }
  v5 = a2[1];
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 0;
  for ( i = (v4 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
              | ((unsigned __int64)(((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))); ; i = (v4 - 1) & v12 )
  {
    v10 = (__int64 *)(v7 + 32LL * i);
    v11 = *v10;
    if ( *v10 == *a2 && v10[1] == v5 )
      return v10 + 2;
    if ( v11 == -4096 )
      break;
    if ( v11 == -8192 && v10[1] == -8192 && !v8 )
      v8 = (__int64 *)(v7 + 32LL * i);
LABEL_9:
    v12 = v6 + i;
    ++v6;
  }
  if ( v10[1] != -4096 )
    goto LABEL_9;
  v14 = *(_DWORD *)(a1 + 16);
  if ( !v8 )
    v8 = v10;
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  v17 = v8;
  if ( 4 * (v14 + 1) < 3 * v4 )
  {
    if ( v4 - *(_DWORD *)(a1 + 20) - v15 > v4 >> 3 )
      goto LABEL_17;
    goto LABEL_24;
  }
LABEL_23:
  v4 *= 2;
LABEL_24:
  sub_2C9A610(a1, v4);
  sub_2C95DC0(a1, a2, &v17);
  v8 = v17;
  v15 = *(_DWORD *)(a1 + 16) + 1;
LABEL_17:
  *(_DWORD *)(a1 + 16) = v15;
  if ( *v8 != -4096 || v8[1] != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v8 = *a2;
  v16 = a2[1];
  v8[2] = 0;
  v8[1] = v16;
  v8[3] = 0;
  return v8 + 2;
}
