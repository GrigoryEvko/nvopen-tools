// Function: sub_CE7A30
// Address: 0xce7a30
//
__int64 __fastcall sub_CE7A30(__int64 a1, __int64 a2)
{
  const void *v3; // r8
  size_t v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rsi
  unsigned int v7; // eax
  _BYTE *v8; // rdi
  unsigned int v9; // r12d
  int v10; // edx
  __int64 v11; // rcx
  _DWORD *v12; // r8
  __int64 v13; // rcx
  _DWORD *v14; // rax
  _DWORD *v15; // rcx
  _BYTE *v17; // [rsp+0h] [rbp-70h] BYREF
  __int64 v18; // [rsp+8h] [rbp-68h]
  _BYTE v19[96]; // [rsp+10h] [rbp-60h] BYREF

  v3 = *(const void **)a2;
  v4 = *(_QWORD *)(a2 + 8);
  v17 = v19;
  v5 = *(_QWORD *)(a1 + 24);
  v18 = 0x1000000000LL;
  v6 = (__int64)v3;
  v7 = sub_CE7920(v5, v3, v4, (__int64)&v17);
  if ( !(_BYTE)v7 )
  {
    v8 = v17;
    goto LABEL_13;
  }
  v8 = v17;
  v9 = v7;
  v10 = *(_DWORD *)(a1 + 32);
  v11 = 4LL * (unsigned int)v18;
  v12 = &v17[v11];
  v6 = v11 >> 2;
  v13 = v11 >> 4;
  if ( !v13 )
  {
    v14 = v17;
LABEL_21:
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v6 != 1 )
          goto LABEL_13;
        goto LABEL_24;
      }
      if ( v10 == *v14 )
        goto LABEL_9;
      ++v14;
    }
    if ( v10 == *v14 )
      goto LABEL_9;
    ++v14;
LABEL_24:
    if ( v10 != *v14 )
      goto LABEL_13;
    goto LABEL_9;
  }
  v14 = v17;
  v15 = &v17[16 * v13];
  while ( v10 != *v14 )
  {
    if ( v10 == v14[1] )
    {
      ++v14;
      break;
    }
    if ( v10 == v14[2] )
    {
      v14 += 2;
      break;
    }
    if ( v10 == v14[3] )
    {
      v14 += 3;
      break;
    }
    v14 += 4;
    if ( v14 == v15 )
    {
      v6 = v12 - v14;
      goto LABEL_21;
    }
  }
LABEL_9:
  if ( v12 != v14 )
  {
    if ( v17 != v19 )
      _libc_free(v17, v6);
    return v9;
  }
LABEL_13:
  if ( v8 != v19 )
    _libc_free(v8, v6);
  return 0;
}
