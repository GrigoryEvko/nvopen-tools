// Function: sub_1F6CD90
// Address: 0x1f6cd90
//
__int64 __fastcall sub_1F6CD90(__int64 a1, char *a2, __int64 a3, unsigned int a4, int a5)
{
  char *v5; // r9
  signed __int64 v6; // r12
  __int64 v7; // rdx
  unsigned int v8; // r10d
  int *v9; // rbx
  int v10; // r14d
  __int64 v11; // r15
  __int64 v12; // r12
  int v13; // ebx
  int v14; // r13d
  unsigned int v15; // ebx
  _BYTE *v16; // rdi
  __int64 v18; // rbx
  unsigned __int64 v19; // r8
  size_t v20; // rdx
  void *dest; // [rsp+0h] [rbp-90h]
  int *v23; // [rsp+18h] [rbp-78h]
  unsigned int v24; // [rsp+24h] [rbp-6Ch]
  int *v25; // [rsp+28h] [rbp-68h]
  void *src; // [rsp+30h] [rbp-60h] BYREF
  __int64 v27; // [rsp+38h] [rbp-58h]
  _BYTE v28[80]; // [rsp+40h] [rbp-50h] BYREF

  v5 = a2;
  v6 = 4 * a3;
  dest = (void *)(a1 + 16);
  if ( a4 == 1 )
  {
    v18 = v6 >> 2;
    *(_QWORD *)a1 = dest;
    *(_QWORD *)(a1 + 8) = 0x800000000LL;
    if ( (unsigned __int64)v6 > 0x20 )
    {
      sub_16CD150(a1, dest, v6 >> 2, 4, a5, (int)a2);
      v5 = a2;
      dest = (void *)(*(_QWORD *)a1 + 4LL * *(unsigned int *)(a1 + 8));
    }
    else if ( !v6 )
    {
LABEL_21:
      *(_DWORD *)(a1 + 8) = v18 + v6;
      return a1;
    }
    memcpy(dest, v5, v6);
    LODWORD(v6) = *(_DWORD *)(a1 + 8);
    goto LABEL_21;
  }
  src = v28;
  v27 = 0x800000000LL;
  v23 = (int *)&a2[v6];
  if ( a2 == &a2[v6] )
  {
    *(_QWORD *)(a1 + 8) = 0x800000000LL;
    *(_QWORD *)a1 = dest;
    return a1;
  }
  v7 = 0;
  v8 = a4;
  v9 = (int *)a2;
  do
  {
    v10 = *v9;
    if ( v8 )
    {
      v25 = v9;
      v11 = v8;
      v12 = 0;
      v13 = v8 * v10;
      do
      {
        v14 = v13 + v12;
        if ( v10 < 0 )
          v14 = -1;
        if ( HIDWORD(v27) <= (unsigned int)v7 )
        {
          v24 = v8;
          sub_16CD150((__int64)&src, v28, 0, 4, -1, (int)v5);
          v7 = (unsigned int)v27;
          v8 = v24;
        }
        ++v12;
        *((_DWORD *)src + v7) = v14;
        v7 = (unsigned int)(v27 + 1);
        LODWORD(v27) = v27 + 1;
      }
      while ( v11 != v12 );
      v9 = v25;
    }
    ++v9;
  }
  while ( v23 != v9 );
  v15 = v7;
  v16 = src;
  *(_QWORD *)a1 = dest;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  if ( (_DWORD)v7 )
  {
    if ( v16 != v28 )
    {
      *(_QWORD *)a1 = v16;
      *(_DWORD *)(a1 + 8) = v7;
      *(_DWORD *)(a1 + 12) = HIDWORD(v27);
      return a1;
    }
    v19 = (unsigned int)v7;
    v20 = 4LL * (unsigned int)v7;
    if ( v15 <= 8
      || (sub_16CD150(a1, dest, v19, 4, v19, (int)v5), v16 = src,
                                                       v20 = 4LL * (unsigned int)v27,
                                                       dest = *(void **)a1,
                                                       v20) )
    {
      memcpy(dest, v16, v20);
      v16 = src;
    }
    *(_DWORD *)(a1 + 8) = v15;
  }
  if ( v16 != v28 )
    _libc_free((unsigned __int64)v16);
  return a1;
}
