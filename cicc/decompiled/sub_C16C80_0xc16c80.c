// Function: sub_C16C80
// Address: 0xc16c80
//
__int64 __fastcall sub_C16C80(__int64 a1, __int64 a2)
{
  unsigned __int64 *v3; // rax
  unsigned __int64 v4; // rdx
  _QWORD *v5; // r8
  unsigned __int64 *v6; // rbx
  __int64 v7; // r14
  unsigned int v8; // r12d
  unsigned __int64 v9; // r12
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  __int64 v14; // rax
  _BYTE *v15; // rdi
  char v16; // al
  void *v17; // r8
  size_t v18; // rdx
  char v19; // al
  int v20; // eax
  _QWORD *v21; // [rsp+0h] [rbp-140h]
  __int64 v22; // [rsp+18h] [rbp-128h] BYREF
  void *src; // [rsp+20h] [rbp-120h] BYREF
  __int64 v24; // [rsp+28h] [rbp-118h]
  _BYTE v25[272]; // [rsp+30h] [rbp-110h] BYREF

  v3 = *(unsigned __int64 **)a2;
  v4 = **(_QWORD **)a2;
  if ( v4 > 0x1C )
  {
    sub_C15F80((__int64 *)&src, 9, "memprof schema invalid");
    v12 = (unsigned __int64)src;
    *(_BYTE *)(a1 + 240) |= 3u;
    *(_QWORD *)a1 = v12 & 0xFFFFFFFFFFFFFFFELL;
  }
  else
  {
    v5 = (_QWORD *)a2;
    v6 = v3 + 1;
    src = v25;
    v24 = 0x1C00000000LL;
    if ( v4 )
    {
      v7 = (__int64)&v3[v4 + 1];
      do
      {
        v9 = *v6++;
        if ( v9 > 0x1B )
        {
          a2 = 9;
          sub_C15F80(&v22, 9, "memprof schema invalid");
          v14 = v22;
          *(_BYTE *)(a1 + 240) |= 3u;
          v15 = src;
          *(_QWORD *)a1 = v14 & 0xFFFFFFFFFFFFFFFELL;
          goto LABEL_11;
        }
        v10 = (unsigned int)v24;
        v11 = (unsigned int)v24 + 1LL;
        if ( v11 > HIDWORD(v24) )
        {
          a2 = (__int64)v25;
          v21 = v5;
          sub_C8D5F0(&src, v25, v11, 8);
          v10 = (unsigned int)v24;
          v5 = v21;
        }
        *((_QWORD *)src + v10) = v9;
        v8 = v24 + 1;
        LODWORD(v24) = v24 + 1;
      }
      while ( v6 != (unsigned __int64 *)v7 );
      *v5 = v6;
      v16 = *(_BYTE *)(a1 + 240);
      v17 = (void *)(a1 + 16);
      *(_QWORD *)a1 = a1 + 16;
      v15 = src;
      *(_BYTE *)(a1 + 240) = v16 & 0xFC | 2;
      *(_QWORD *)(a1 + 8) = 0x1C00000000LL;
      if ( v8 )
      {
        if ( v15 != v25 )
        {
          v20 = HIDWORD(v24);
          *(_QWORD *)a1 = v15;
          *(_DWORD *)(a1 + 8) = v8;
          *(_DWORD *)(a1 + 12) = v20;
          return a1;
        }
        v18 = 8LL * v8;
        if ( v8 <= 0x1C
          || (a2 = a1 + 16,
              sub_C8D5F0(a1, a1 + 16, v8, 8),
              v17 = *(void **)a1,
              v15 = src,
              (v18 = 8LL * (unsigned int)v24) != 0) )
        {
          a2 = (__int64)v15;
          memcpy(v17, v15, v18);
          v15 = src;
        }
        *(_DWORD *)(a1 + 8) = v8;
      }
LABEL_11:
      if ( v15 != v25 )
        _libc_free(v15, a2);
    }
    else
    {
      *(_QWORD *)a2 = v6;
      v19 = *(_BYTE *)(a1 + 240);
      *(_QWORD *)(a1 + 8) = 0x1C00000000LL;
      *(_BYTE *)(a1 + 240) = v19 & 0xFC | 2;
      *(_QWORD *)a1 = a1 + 16;
    }
  }
  return a1;
}
