// Function: sub_31BE2F0
// Address: 0x31be2f0
//
__int64 __fastcall sub_31BE2F0(__int64 a1)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 result; // rax
  unsigned int v6; // r12d
  __int64 v7; // rbx
  _QWORD *v8; // rdi
  unsigned int v9; // ebx
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 i; // rdx
  unsigned int v13; // [rsp+Ch] [rbp-34h]

  v2 = *(unsigned int *)(a1 + 24);
  v13 = *(_DWORD *)(a1 + 16);
  if ( !(_DWORD)v2 )
  {
    result = v13;
    if ( !v13 )
      goto LABEL_30;
    v6 = v13 - 1;
    if ( v13 == 1 )
    {
      v8 = *(_QWORD **)(a1 + 8);
      LODWORD(v7) = 64;
LABEL_17:
      sub_C7D6A0((__int64)v8, 40 * v2, 8);
      v9 = 4 * (int)v7 / 3u;
      v10 = ((((((((((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2) | (v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 4)
               | (((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2)
               | (v9 + 1)
               | ((unsigned __int64)(v9 + 1) >> 1)) >> 8)
             | (((((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2) | (v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 4)
             | (((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2)
             | (v9 + 1)
             | ((unsigned __int64)(v9 + 1) >> 1)) >> 16)
           | (((((((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2) | (v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 4)
             | (((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2)
             | (v9 + 1)
             | ((unsigned __int64)(v9 + 1) >> 1)) >> 8)
           | (((((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2) | (v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 4)
           | (((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2)
           | (v9 + 1)
           | ((unsigned __int64)(v9 + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 24) = v10;
      result = sub_C7D670(40 * v10, 8);
      v11 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = result;
      for ( i = result + 40 * v11; i != result; result += 40 )
      {
        if ( result )
          *(_QWORD *)result = -4096;
      }
      return result;
    }
    LODWORD(result) = 0;
    goto LABEL_14;
  }
  v3 = *(_QWORD *)(a1 + 8);
  v4 = v3 + 40LL * (unsigned int)v2;
  do
  {
    if ( *(_QWORD *)v3 != -4096 && *(_QWORD *)v3 != -8192 )
      sub_C7D6A0(*(_QWORD *)(v3 + 16), 16LL * *(unsigned int *)(v3 + 32), 8);
    v3 += 40;
  }
  while ( v4 != v3 );
  result = *(unsigned int *)(a1 + 24);
  if ( !v13 )
  {
    if ( (_DWORD)result )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 8), 40LL * (unsigned int)v2, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
LABEL_30:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  v7 = 64;
  v6 = v13 - 1;
  if ( v13 != 1 )
  {
LABEL_14:
    _BitScanReverse(&v6, v6);
    v7 = (unsigned int)(1 << (33 - (v6 ^ 0x1F)));
    if ( (int)v7 < 64 )
      v7 = 64;
  }
  v8 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)v7 != (_DWORD)result )
    goto LABEL_17;
  *(_QWORD *)(a1 + 16) = 0;
  result = (__int64)&v8[5 * v7];
  do
  {
    if ( v8 )
      *v8 = -4096;
    v8 += 5;
  }
  while ( (_QWORD *)result != v8 );
  return result;
}
