// Function: sub_160FB80
// Address: 0x160fb80
//
__int64 __fastcall sub_160FB80(__int64 a1)
{
  __int64 v2; // rbx
  int v3; // eax
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 i; // rdx
  unsigned int v7; // ecx
  _QWORD *v8; // rdi
  unsigned int v9; // eax
  int v10; // eax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  int v13; // r14d
  __int64 v14; // r13
  __int64 v15; // rdx
  __int64 j; // rdx

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 8) - 8LL);
  v3 = *(_DWORD *)(v2 + 240);
  ++*(_QWORD *)(v2 + 224);
  if ( !v3 )
  {
    result = *(unsigned int *)(v2 + 244);
    if ( !(_DWORD)result )
      goto LABEL_7;
    v5 = *(unsigned int *)(v2 + 248);
    if ( (unsigned int)v5 > 0x40 )
    {
      result = j___libc_free_0(*(_QWORD *)(v2 + 232));
      *(_QWORD *)(v2 + 232) = 0;
      *(_QWORD *)(v2 + 240) = 0;
      *(_DWORD *)(v2 + 248) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v7 = 4 * v3;
  v5 = *(unsigned int *)(v2 + 248);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v7 = 64;
  if ( v7 >= (unsigned int)v5 )
  {
LABEL_4:
    result = *(_QWORD *)(v2 + 232);
    for ( i = result + 16 * v5; i != result; result += 16 )
      *(_QWORD *)result = -4;
    *(_QWORD *)(v2 + 240) = 0;
    goto LABEL_7;
  }
  v8 = *(_QWORD **)(v2 + 232);
  v9 = v3 - 1;
  if ( !v9 )
  {
    v14 = 2048;
    v13 = 128;
LABEL_16:
    j___libc_free_0(v8);
    *(_DWORD *)(v2 + 248) = v13;
    result = sub_22077B0(v14);
    v15 = *(unsigned int *)(v2 + 248);
    *(_QWORD *)(v2 + 240) = 0;
    *(_QWORD *)(v2 + 232) = result;
    for ( j = result + 16 * v15; j != result; result += 16 )
    {
      if ( result )
        *(_QWORD *)result = -4;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v9, v9);
  v10 = 1 << (33 - (v9 ^ 0x1F));
  if ( v10 < 64 )
    v10 = 64;
  if ( (_DWORD)v5 != v10 )
  {
    v11 = (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
        | (4 * v10 / 3u + 1)
        | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)
        | (((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
          | (4 * v10 / 3u + 1)
          | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4);
    v12 = (v11 >> 8) | v11;
    v13 = (v12 | (v12 >> 16)) + 1;
    v14 = 16 * ((v12 | (v12 >> 16)) + 1);
    goto LABEL_16;
  }
  *(_QWORD *)(v2 + 240) = 0;
  result = (__int64)&v8[2 * (unsigned int)v5];
  do
  {
    if ( v8 )
      *v8 = -4;
    v8 += 2;
  }
  while ( (_QWORD *)result != v8 );
LABEL_7:
  *(_QWORD *)(v2 + 168) = 0;
  *(_QWORD *)(v2 + 176) = 0;
  *(_QWORD *)(v2 + 184) = 0;
  *(_QWORD *)(v2 + 192) = 0;
  *(_QWORD *)(v2 + 200) = 0;
  *(_QWORD *)(v2 + 208) = 0;
  *(_QWORD *)(v2 + 216) = 0;
  *(_QWORD *)(a1 + 8) -= 8LL;
  return result;
}
