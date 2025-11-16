// Function: sub_30B0A30
// Address: 0x30b0a30
//
bool __fastcall sub_30B0A30(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  _QWORD *v5; // rbx
  _QWORD *i; // r13
  __int64 v7; // r15
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 *v12; // r10
  __int64 *v13; // r13
  __int64 v14; // rdi
  int v15; // r9d
  __int64 v16; // rdx
  void *v17; // r8
  size_t v18; // r11
  _BYTE *v19; // rdi
  int v20; // [rsp+0h] [rbp-A0h]
  __int64 v21; // [rsp+8h] [rbp-98h]
  int v22; // [rsp+10h] [rbp-90h]
  void *v23; // [rsp+10h] [rbp-90h]
  __int64 *v24; // [rsp+18h] [rbp-88h]
  void *src; // [rsp+20h] [rbp-80h] BYREF
  __int64 v26; // [rsp+28h] [rbp-78h]
  _BYTE v27[112]; // [rsp+30h] [rbp-70h] BYREF

  v4 = *(_DWORD *)(a1 + 56);
  if ( (unsigned int)(v4 - 1) > 1 )
  {
    if ( v4 != 3 )
      BUG();
    v12 = *(__int64 **)(a1 + 64);
    v13 = &v12[*(unsigned int *)(a1 + 72)];
    if ( v13 != v12 )
    {
      v24 = *(__int64 **)(a1 + 64);
      do
      {
        v14 = *v24;
        src = v27;
        v26 = 0x800000000LL;
        sub_30B0A30(v14, a2, &src);
        v15 = v26;
        v16 = *(unsigned int *)(a3 + 8);
        v17 = src;
        v18 = 8LL * (unsigned int)v26;
        if ( v16 + (unsigned __int64)(unsigned int)v26 > *(unsigned int *)(a3 + 12) )
        {
          v20 = v26;
          v21 = 8LL * (unsigned int)v26;
          v23 = src;
          sub_C8D5F0(a3, (const void *)(a3 + 16), v16 + (unsigned int)v26, 8u, (__int64)src, (unsigned int)v26);
          v16 = *(unsigned int *)(a3 + 8);
          v15 = v20;
          v18 = v21;
          v17 = v23;
        }
        if ( v18 )
        {
          v22 = v15;
          memcpy((void *)(*(_QWORD *)a3 + 8 * v16), v17, v18);
          LODWORD(v16) = *(_DWORD *)(a3 + 8);
          v15 = v22;
        }
        v19 = src;
        *(_DWORD *)(a3 + 8) = v15 + v16;
        if ( v19 != v27 )
          _libc_free((unsigned __int64)v19);
        ++v24;
      }
      while ( v13 != v24 );
    }
  }
  else
  {
    v5 = *(_QWORD **)(a1 + 64);
    for ( i = &v5[*(unsigned int *)(a1 + 72)]; i != v5; ++*(_DWORD *)(a3 + 8) )
    {
      while ( 1 )
      {
        v7 = *v5;
        if ( (*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD))a2)(*(_QWORD *)(a2 + 8), *v5) )
          break;
        if ( i == ++v5 )
          return *(_DWORD *)(a3 + 8) != 0;
      }
      v10 = *(unsigned int *)(a3 + 8);
      if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v10 + 1, 8u, v8, v9);
        v10 = *(unsigned int *)(a3 + 8);
      }
      ++v5;
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v10) = v7;
    }
  }
  return *(_DWORD *)(a3 + 8) != 0;
}
