// Function: sub_2FC8EE0
// Address: 0x2fc8ee0
//
__int64 __fastcall sub_2FC8EE0(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v3; // r13
  __int64 v4; // r14
  __int64 v5; // rax
  void (__fastcall *v6)(_QWORD *, __int64, _QWORD); // rbx
  __int64 v7; // rax
  void (*v8)(); // rax
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // rbx
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  int v14; // eax
  unsigned int v15; // ecx
  __int64 v16; // rdx
  __int64 j; // rdx
  unsigned int v18; // eax
  _QWORD *v19; // rdi
  int v20; // ebx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 i; // rdx
  const char *v25; // [rsp-58h] [rbp-58h] BYREF
  char v26; // [rsp-38h] [rbp-38h]
  char v27; // [rsp-37h] [rbp-37h]

  result = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(a1 + 8) != result )
  {
    v3 = *(_QWORD **)(*(_QWORD *)a1 + 224LL);
    v4 = v3[1];
    (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*v3 + 176LL))(v3, *(_QWORD *)(*(_QWORD *)(v4 + 168) + 440LL), 0);
    v5 = *v3;
    v27 = 1;
    v6 = *(void (__fastcall **)(_QWORD *, __int64, _QWORD))(v5 + 208);
    v26 = 3;
    v25 = "__LLVM_StackMaps";
    v7 = sub_E6C460(v4, &v25);
    v6(v3, v7, 0);
    sub_2FC8A70(a1, (__int64)v3);
    sub_2FC8B20(a1, (__int64)v3);
    sub_2FC8BA0(a1, (__int64)v3);
    sub_2FC8C00(a1, (__int64)v3);
    v8 = *(void (**)())(*v3 + 160LL);
    if ( v8 != nullsub_99 )
      ((void (__fastcall *)(_QWORD *))v8)(v3);
    v9 = *(_QWORD *)(a1 + 8);
    v10 = *(_QWORD *)(a1 + 16);
    if ( v9 != v10 )
    {
      v11 = *(_QWORD *)(a1 + 8);
      do
      {
        v12 = *(_QWORD *)(v11 + 128);
        if ( v12 != v11 + 144 )
          _libc_free(v12);
        v13 = *(_QWORD *)(v11 + 16);
        if ( v13 != v11 + 32 )
          _libc_free(v13);
        v11 += 192;
      }
      while ( v10 != v11 );
      *(_QWORD *)(a1 + 16) = v9;
    }
    v14 = *(_DWORD *)(a1 + 48);
    ++*(_QWORD *)(a1 + 32);
    if ( v14 )
    {
      v15 = 4 * v14;
      v16 = *(unsigned int *)(a1 + 56);
      if ( (unsigned int)(4 * v14) < 0x40 )
        v15 = 64;
      if ( (unsigned int)v16 <= v15 )
        goto LABEL_16;
      v18 = v14 - 1;
      if ( v18 )
      {
        _BitScanReverse(&v18, v18);
        v19 = *(_QWORD **)(a1 + 40);
        v20 = 1 << (33 - (v18 ^ 0x1F));
        if ( v20 < 64 )
          v20 = 64;
        if ( (_DWORD)v16 == v20 )
        {
          *(_QWORD *)(a1 + 48) = 0;
          result = (__int64)&v19[2 * (unsigned int)v16];
          do
          {
            if ( v19 )
              *v19 = -1;
            v19 += 2;
          }
          while ( (_QWORD *)result != v19 );
          goto LABEL_19;
        }
      }
      else
      {
        v19 = *(_QWORD **)(a1 + 40);
        v20 = 64;
      }
      sub_C7D6A0((__int64)v19, 16LL * (unsigned int)v16, 8);
      v21 = ((((((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
               | (4 * v20 / 3u + 1)
               | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
             | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
             | (4 * v20 / 3u + 1)
             | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
             | (4 * v20 / 3u + 1)
             | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
           | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 16;
      v22 = (v21
           | (((((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
               | (4 * v20 / 3u + 1)
               | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
             | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
             | (4 * v20 / 3u + 1)
             | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
             | (4 * v20 / 3u + 1)
             | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
           | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 56) = v22;
      result = sub_C7D670(16 * v22, 8);
      v23 = *(unsigned int *)(a1 + 56);
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 40) = result;
      for ( i = result + 16 * v23; i != result; result += 16 )
      {
        if ( result )
          *(_QWORD *)result = -1;
      }
    }
    else
    {
      result = *(unsigned int *)(a1 + 52);
      if ( (_DWORD)result )
      {
        v16 = *(unsigned int *)(a1 + 56);
        if ( (unsigned int)v16 <= 0x40 )
        {
LABEL_16:
          result = *(_QWORD *)(a1 + 40);
          for ( j = result + 16 * v16; j != result; result += 16 )
            *(_QWORD *)result = -1;
          *(_QWORD *)(a1 + 48) = 0;
          goto LABEL_19;
        }
        result = sub_C7D6A0(*(_QWORD *)(a1 + 40), 16LL * (unsigned int)v16, 8);
        *(_QWORD *)(a1 + 40) = 0;
        *(_QWORD *)(a1 + 48) = 0;
        *(_DWORD *)(a1 + 56) = 0;
      }
    }
LABEL_19:
    *(_DWORD *)(a1 + 72) = 0;
  }
  return result;
}
