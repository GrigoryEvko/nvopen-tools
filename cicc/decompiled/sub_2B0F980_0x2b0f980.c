// Function: sub_2B0F980
// Address: 0x2b0f980
//
void __fastcall sub_2B0F980(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  unsigned int v8; // eax
  bool v9; // cf
  char **v10; // r12
  __int64 v11; // rdx
  __int64 v12; // r12
  __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // [rsp+8h] [rbp-78h]
  char *v17; // [rsp+10h] [rbp-70h] BYREF
  __int64 v18; // [rsp+18h] [rbp-68h]
  _BYTE v19[96]; // [rsp+20h] [rbp-60h] BYREF

  if ( a1 != a2 )
  {
    v6 = a1 + 64;
    if ( a2 != a1 + 64 )
    {
      do
      {
        v8 = *(_DWORD *)(v6 + 8);
        v9 = *(_DWORD *)(a1 + 8) < v8;
        v17 = v19;
        v10 = (char **)v6;
        if ( v9 )
        {
          v11 = 0x600000000LL;
          v18 = 0x600000000LL;
          if ( v8 )
            sub_2B0F6D0((__int64)&v17, (char **)v6, 0x600000000LL, a4, a5, a6);
          v16 = v6 + 64;
          v12 = (v6 - a1) >> 6;
          if ( v6 - a1 > 0 )
          {
            do
            {
              v13 = v6;
              v6 -= 64;
              sub_2B0F6D0(v13, (char **)v6, v11, a4, a5, a6);
              --v12;
            }
            while ( v12 );
          }
          sub_2B0F6D0(a1, &v17, v11, a4, a5, a6);
          if ( v17 != v19 )
            _libc_free((unsigned __int64)v17);
        }
        else
        {
          v14 = 0x600000000LL;
          v18 = 0x600000000LL;
          if ( v8 )
          {
            sub_2B0F6D0((__int64)&v17, (char **)v6, a3, 0x600000000LL, a5, a6);
            if ( (unsigned int)v18 > *(_DWORD *)(v6 - 56) )
            {
              do
              {
                v15 = (__int64)v10;
                v10 -= 8;
                sub_2B0F6D0(v15, v10, a3, v14, a5, a6);
              }
              while ( *((_DWORD *)v10 - 14) < (unsigned int)v18 );
            }
          }
          sub_2B0F6D0((__int64)v10, &v17, a3, v14, a5, a6);
          if ( v17 != v19 )
            _libc_free((unsigned __int64)v17);
          v16 = v6 + 64;
        }
        v6 = v16;
      }
      while ( a2 != v16 );
    }
  }
}
