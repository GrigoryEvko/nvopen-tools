// Function: sub_2F63B30
// Address: 0x2f63b30
//
__int64 __fastcall sub_2F63B30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // edx
  __int64 v8; // rax
  _DWORD *v9; // rdi
  _DWORD *v10; // r12
  unsigned int v11; // ecx
  __int64 result; // rax
  _DWORD *v13; // rbx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rdi
  unsigned int v20; // edx
  char v21; // cl
  _DWORD *v22; // rdx
  __int64 v23; // rbx
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rax
  __int64 v26; // rdx
  __int64 i; // rdx
  unsigned __int64 v28[2]; // [rsp+0h] [rbp-80h] BYREF
  _BYTE v29[112]; // [rsp+10h] [rbp-70h] BYREF

  v7 = *(_DWORD *)(a1 + 904);
  v8 = *(unsigned int *)(a1 + 912);
  v9 = *(_DWORD **)(a1 + 896);
  v10 = &v9[v8];
  if ( v7 )
  {
    v13 = v9;
    if ( v10 != v9 )
    {
      while ( *v13 > 0xFFFFFFFD )
      {
        if ( ++v13 == v10 )
          goto LABEL_2;
      }
      if ( v13 == v10 )
      {
        ++*(_QWORD *)(a1 + 888);
        goto LABEL_4;
      }
LABEL_19:
      v14 = *(_QWORD *)(a1 + 40);
      v15 = *v13 & 0x7FFFFFFF;
      if ( (unsigned int)v15 < *(_DWORD *)(v14 + 160) )
      {
        v16 = *(_QWORD *)(*(_QWORD *)(v14 + 152) + 8 * v15);
        if ( v16 )
        {
          v17 = *(_QWORD *)(*(_QWORD *)(v14 + 152) + 8 * v15);
          if ( (unsigned __int8)sub_2E168A0((_QWORD *)v14, v16, a1 + 760, a4, a5, a6) )
          {
            v19 = *(_QWORD *)(a1 + 40);
            v17 = v16;
            v28[0] = (unsigned __int64)v29;
            v28[1] = 0x800000000LL;
            sub_2E15100(v19, v16, (__int64)v28);
            if ( (_BYTE *)v28[0] != v29 )
              _libc_free(v28[0]);
          }
          v18 = *(unsigned int *)(a1 + 768);
          if ( (_DWORD)v18 )
            sub_2F62AC0(a1, v17, v18, a4, a5, a6);
        }
      }
      while ( ++v13 != v10 )
      {
        if ( *v13 <= 0xFFFFFFFD )
        {
          if ( v13 != v10 )
            goto LABEL_19;
          break;
        }
      }
      v7 = *(_DWORD *)(a1 + 904);
    }
  }
LABEL_2:
  ++*(_QWORD *)(a1 + 888);
  if ( !v7 )
  {
    result = *(unsigned int *)(a1 + 908);
    if ( !(_DWORD)result )
      return result;
    result = *(unsigned int *)(a1 + 912);
    if ( (unsigned int)result > 0x40 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 896), 4 * result, 4);
      *(_QWORD *)(a1 + 896) = 0;
      *(_QWORD *)(a1 + 904) = 0;
      *(_DWORD *)(a1 + 912) = 0;
      return result;
    }
    v9 = *(_DWORD **)(a1 + 896);
    goto LABEL_7;
  }
  v9 = *(_DWORD **)(a1 + 896);
LABEL_4:
  v11 = 4 * v7;
  result = *(unsigned int *)(a1 + 912);
  if ( (unsigned int)(4 * v7) < 0x40 )
    v11 = 64;
  if ( (unsigned int)result <= v11 )
  {
LABEL_7:
    if ( 4LL * (unsigned int)result )
      result = (__int64)memset(v9, 255, 4LL * (unsigned int)result);
    *(_QWORD *)(a1 + 904) = 0;
    return result;
  }
  v20 = v7 - 1;
  if ( v20 )
  {
    _BitScanReverse(&v20, v20);
    v21 = 33 - (v20 ^ 0x1F);
    v22 = v9;
    v23 = (unsigned int)(1 << v21);
    if ( (int)v23 < 64 )
      v23 = 64;
    if ( (_DWORD)v23 == (_DWORD)result )
    {
      *(_QWORD *)(a1 + 904) = 0;
      result = (__int64)&v9[v23];
      do
      {
        if ( v22 )
          *v22 = -1;
        ++v22;
      }
      while ( (_DWORD *)result != v22 );
      return result;
    }
  }
  else
  {
    LODWORD(v23) = 64;
  }
  sub_C7D6A0((__int64)v9, 4 * result, 4);
  v24 = ((((((((4 * (int)v23 / 3u + 1) | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v23 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v23 / 3u + 1) | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v23 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1)) >> 8)
       | (((((4 * (int)v23 / 3u + 1) | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v23 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1)) >> 4)
       | (((4 * (int)v23 / 3u + 1) | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1)) >> 2)
       | (4 * (int)v23 / 3u + 1)
       | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1)) >> 16;
  v25 = (v24
       | (((((((4 * (int)v23 / 3u + 1) | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v23 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v23 / 3u + 1) | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v23 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1)) >> 8)
       | (((((4 * (int)v23 / 3u + 1) | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v23 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1)) >> 4)
       | (((4 * (int)v23 / 3u + 1) | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1)) >> 2)
       | (4 * (int)v23 / 3u + 1)
       | ((unsigned __int64)(4 * (int)v23 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 912) = v25;
  result = sub_C7D670(4 * v25, 4);
  v26 = *(unsigned int *)(a1 + 912);
  *(_QWORD *)(a1 + 904) = 0;
  *(_QWORD *)(a1 + 896) = result;
  for ( i = result + 4 * v26; i != result; result += 4 )
  {
    if ( result )
      *(_DWORD *)result = -1;
  }
  return result;
}
