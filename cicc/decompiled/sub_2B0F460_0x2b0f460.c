// Function: sub_2B0F460
// Address: 0x2b0f460
//
void __fastcall sub_2B0F460(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 *v9; // rbx
  __int64 v10; // r14
  unsigned int v11; // eax
  bool v12; // cc
  __int64 v13; // rcx
  __int64 v14; // r14
  __int64 v15; // rdx
  unsigned __int64 v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // [rsp+0h] [rbp-A0h]
  __int64 v23; // [rsp+10h] [rbp-90h]
  __int64 v24; // [rsp+20h] [rbp-80h]
  char *v25; // [rsp+28h] [rbp-78h] BYREF
  __int64 v26; // [rsp+30h] [rbp-70h]
  _BYTE v27[104]; // [rsp+38h] [rbp-68h] BYREF

  if ( a1 != a2 && a2 != a1 + 72 )
  {
    v7 = a1 + 80;
    v22 = a1 + 8;
    do
    {
      v8 = *(_QWORD *)(v7 - 8);
      v9 = (__int64 *)(v7 - 8);
      v10 = v7;
      v11 = *(_DWORD *)(v7 + 8);
      v12 = v11 <= *(_DWORD *)(a1 + 16);
      v25 = v27;
      v13 = 0xC00000000LL;
      v24 = v8;
      v26 = 0xC00000000LL;
      if ( v12 )
      {
        if ( v11 )
        {
          sub_2B0D090((__int64)&v25, (char **)v7, v8, 0xC00000000LL, a5, a6);
          if ( *(_DWORD *)(v7 - 64) < (unsigned int)v26 )
          {
            do
            {
              v20 = *(_QWORD *)(v10 - 80);
              v21 = v10;
              v9 = (__int64 *)(v10 - 80);
              v10 -= 72;
              *(_QWORD *)(v10 + 64) = v20;
              sub_2B0D090(v21, (char **)v10, v19, v13, a5, a6);
            }
            while ( (unsigned int)v26 > *(_DWORD *)(v10 - 64) );
          }
          v8 = v24;
        }
        *v9 = v8;
        sub_2B0D090(v10, &v25, v8, v13, a5, a6);
        if ( v25 != v27 )
          _libc_free((unsigned __int64)v25);
        v23 = v7 + 64;
      }
      else
      {
        if ( v11 )
          sub_2B0D090((__int64)&v25, (char **)v7, v8, 0xC00000000LL, a5, a6);
        v14 = v7;
        v23 = v7 + 64;
        v15 = (__int64)v9 - a1;
        v16 = 0x8E38E38E38E38E39LL * (((__int64)v9 - a1) >> 3);
        if ( v15 > 0 )
        {
          do
          {
            v17 = *(_QWORD *)(v14 - 80);
            v18 = v14;
            v14 -= 72;
            *(_QWORD *)(v14 + 64) = v17;
            sub_2B0D090(v18, (char **)v14, v17, v13, a5, a6);
            --v16;
          }
          while ( v16 );
        }
        *(_QWORD *)a1 = v24;
        sub_2B0D090(v22, &v25, v15, v13, a5, a6);
        if ( v25 != v27 )
          _libc_free((unsigned __int64)v25);
      }
      v7 += 72;
    }
    while ( a2 != v23 );
  }
}
