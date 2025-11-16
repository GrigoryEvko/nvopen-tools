// Function: sub_22AD930
// Address: 0x22ad930
//
void __fastcall sub_22AD930(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char **v7; // r12
  __int64 v8; // rdx
  char *v9; // rbx
  char **v10; // r14
  bool v11; // cc
  int v12; // eax
  __int64 v13; // rcx
  __int64 v14; // r14
  __int64 v15; // rdx
  unsigned __int64 v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rdi
  unsigned int v19; // eax
  __int64 v20; // rdi
  __int64 v21; // [rsp+0h] [rbp-A0h]
  char *v22; // [rsp+10h] [rbp-90h]
  unsigned int v23; // [rsp+20h] [rbp-80h]
  char *v24[2]; // [rsp+28h] [rbp-78h] BYREF
  _BYTE v25[104]; // [rsp+38h] [rbp-68h] BYREF

  if ( a1 != a2 && a2 != a1 + 72 )
  {
    v7 = (char **)(a1 + 80);
    v21 = (__int64)(a1 + 8);
    do
    {
      v8 = *((unsigned int *)v7 - 2);
      v9 = (char *)(v7 - 1);
      v10 = v7;
      v11 = (unsigned int)v8 <= *(_DWORD *)a1;
      v12 = *((_DWORD *)v7 + 2);
      v24[0] = v25;
      v13 = 0xC00000000LL;
      v23 = v8;
      v24[1] = (char *)0xC00000000LL;
      if ( v11 )
      {
        if ( v12 )
        {
          sub_22AD4A0((__int64)v24, v7, v8, 0xC00000000LL, a5, a6);
          v8 = v23;
        }
        v19 = *((_DWORD *)v7 - 20);
        if ( (unsigned int)v8 > v19 )
        {
          do
          {
            *((_DWORD *)v10 - 2) = v19;
            v20 = (__int64)v10;
            v9 = (char *)(v10 - 10);
            v10 -= 9;
            sub_22AD4A0(v20, v10, v8, v13, a5, a6);
            v8 = v23;
            v19 = *((_DWORD *)v10 - 20);
          }
          while ( v23 > v19 );
        }
        *(_DWORD *)v9 = v8;
        sub_22AD4A0((__int64)v10, v24, v8, v13, a5, a6);
        if ( v24[0] != v25 )
          _libc_free((unsigned __int64)v24[0]);
        v22 = (char *)(v7 + 8);
      }
      else
      {
        if ( v12 )
          sub_22AD4A0((__int64)v24, v7, v8, 0xC00000000LL, a5, a6);
        v14 = (__int64)v7;
        v22 = (char *)(v7 + 8);
        v15 = v9 - a1;
        v16 = 0x8E38E38E38E38E39LL * ((v9 - a1) >> 3);
        if ( v15 > 0 )
        {
          do
          {
            v17 = *(unsigned int *)(v14 - 80);
            v18 = v14;
            v14 -= 72;
            *(_DWORD *)(v14 + 64) = v17;
            sub_22AD4A0(v18, (char **)v14, v17, v13, a5, a6);
            --v16;
          }
          while ( v16 );
        }
        *(_DWORD *)a1 = v23;
        sub_22AD4A0(v21, v24, v15, v13, a5, a6);
        if ( v24[0] != v25 )
          _libc_free((unsigned __int64)v24[0]);
      }
      v7 += 9;
    }
    while ( a2 != v22 );
  }
}
