// Function: sub_C88640
// Address: 0xc88640
//
unsigned __int64 __fastcall sub_C88640(_QWORD *a1, __int64 *a2)
{
  int v3; // r11d
  unsigned __int64 v4; // r15
  unsigned __int64 v5; // rcx
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // r9
  int *v8; // r13
  int *v9; // r14
  int v10; // ebx
  unsigned int v11; // r10d
  _DWORD *v12; // rax
  unsigned __int64 v13; // r10
  unsigned __int64 v14; // r12
  _DWORD *v15; // r11
  _DWORD *v16; // r14
  unsigned int v17; // edx
  int v18; // ebx
  int v19; // ebx
  __int64 v20; // rdx
  bool v21; // di
  unsigned __int64 result; // rax
  __int64 v23; // [rsp+0h] [rbp-9F0h]
  _DWORD v24[634]; // [rsp+8h] [rbp-9E8h] BYREF

  v3 = -1953789045;
  memset(v24, 0x8Bu, 0x9C0u);
  v23 = *a2;
  v4 = (a2[1] - *a2) >> 2;
  v5 = v4 + 1;
  if ( v4 + 1 < 0x270 )
    v5 = 624;
  v6 = 0;
  do
  {
    while ( 1 )
    {
      v7 = v6 % 0x270;
      v8 = &v24[v6 % 0x270];
      v9 = &v24[(v6 + 306) % 0x270];
      v10 = *v9;
      v11 = 1664525 * (((v10 ^ *v8 ^ (unsigned int)v3) >> 27) ^ v10 ^ *v8 ^ v3);
      v12 = &v24[(v6 + 317) % 0x270];
      if ( v6 )
        break;
      v3 = v11 + v4;
      v6 = 1;
      *v9 = v10 + v11;
      *v12 += v11 + v4;
      *v8 = v11 + v4;
    }
    v3 = v11 + v7;
    if ( v4 >= v6 )
      v3 = v11 + *(_DWORD *)(v23 + 4 * v6 - 4) + v7;
    ++v6;
    *v9 = v10 + v11;
    *v12 += v3;
    *v8 = v3;
  }
  while ( v5 > v6 );
  if ( v5 + 624 >= v5 )
  {
    v13 = v5 + 1;
    v14 = v5 + (-(__int64)(v5 + 624 < v5 + 1) & 0xFFFFFFFFFFFFFD91LL) + 624;
    while ( 1 )
    {
      v15 = &v24[v5 % 0x270];
      v16 = &v24[(v5 + 306) % 0x270];
      v17 = *v15 + *v16 + v24[(v5 - 1) % 0x270];
      v18 = 1566083941 * (v17 ^ (v17 >> 27));
      *v16 ^= v18;
      v19 = v18 - v5 % 0x270;
      v24[(v5 + 317) % 0x270] ^= v19;
      v5 = v13;
      *v15 = v19;
      if ( v14 == v13 )
        break;
      ++v13;
    }
  }
  v20 = 0;
  v21 = 1;
  while ( 1 )
  {
    result = (unsigned int)v24[2 * v20] + ((unsigned __int64)(unsigned int)v24[2 * v20 + 1] << 32);
    a1[v20] = result;
    if ( !v21 )
      break;
    v21 = result == 0;
    if ( v20 )
      break;
    v21 = (*a1 & 0xFFFFFFFF80000000LL) == 0;
LABEL_14:
    ++v20;
  }
  if ( v20 != 311 )
    goto LABEL_14;
  if ( v21 )
  {
    result = 0x8000000000000000LL;
    *a1 = 0x8000000000000000LL;
  }
  a1[312] = 312;
  return result;
}
