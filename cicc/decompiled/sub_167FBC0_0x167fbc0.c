// Function: sub_167FBC0
// Address: 0x167fbc0
//
void __fastcall sub_167FBC0(__int64 a1, __int64 a2)
{
  char *v3; // rsi
  __int64 v4; // rbx
  const char *v5; // rdx
  const char *v6; // rax
  char *v7; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8; // [rsp+8h] [rbp-28h]
  _BYTE v9[32]; // [rsp+10h] [rbp-20h] BYREF

  v3 = v9;
  v4 = *(_QWORD *)(a1 + 32);
  v7 = v9;
  v8 = 0;
  if ( v4 )
  {
    sub_16CD150(&v7, v9, v4, 1);
    v3 = v7;
    v5 = &v7[v4];
    v6 = &v7[(unsigned int)v8];
    if ( v6 != &v7[v4] )
    {
      do
      {
        if ( v6 )
          *v6 = 0;
        ++v6;
      }
      while ( v5 != v6 );
      v3 = v7;
    }
    LODWORD(v8) = v4;
  }
  sub_167FAF0(a1, v3);
  sub_16E7EE0(a2, v7, (unsigned int)v8);
  if ( v7 != v9 )
    _libc_free((unsigned __int64)v7);
}
