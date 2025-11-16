// Function: sub_2900790
// Address: 0x2900790
//
unsigned __int64 __fastcall sub_2900790(__int64 a1, char a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // r12
  __int64 v4; // rax
  const char *v5; // r15
  __int64 *v6; // rbx
  __int64 v7; // rax
  int i; // esi
  __int64 *v9; // r15
  const void *v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r12
  int v14; // eax
  __int64 v15; // r13
  int v16; // r15d
  int k; // r13d
  __int64 v18; // rax
  __int64 j; // [rsp+20h] [rbp-110h]
  unsigned __int64 v21; // [rsp+28h] [rbp-108h] BYREF
  __int64 v22; // [rsp+30h] [rbp-100h] BYREF
  __int64 v23; // [rsp+38h] [rbp-F8h] BYREF
  _BYTE v24[8]; // [rsp+40h] [rbp-F0h] BYREF
  char *v25; // [rsp+48h] [rbp-E8h]
  char v26; // [rsp+58h] [rbp-D8h] BYREF
  __int64 v27; // [rsp+A0h] [rbp-90h] BYREF
  char *v28; // [rsp+A8h] [rbp-88h]
  char v29; // [rsp+B8h] [rbp-78h] BYREF

  v3 = a3;
  v4 = *(_QWORD *)(a1 + 72);
  v21 = a3;
  v22 = v4;
  if ( v4 )
  {
    v5 = "\\";
    v6 = (__int64 *)sub_BD5C60(a1);
    v7 = sub_A74680(&v22);
    sub_A74940((__int64)v24, (__int64)v6, v7);
    for ( i = 92; ; i = *(_DWORD *)v5 )
    {
      v5 += 4;
      sub_A77390((__int64)v24, i);
      if ( v5 == "<preserve-cfg>" )
        break;
    }
    v23 = sub_A74680(&v22);
    v9 = (__int64 *)sub_A73280(&v23);
    for ( j = sub_A73290(&v23); (__int64 *)j != v9; ++v9 )
    {
      v13 = *v9;
      if ( (unsigned __int8)sub_3145AF0(*v9) )
      {
        v27 = v13;
        if ( sub_A71840((__int64)&v27) )
        {
          v11 = (const void *)sub_A71FD0(&v27);
          sub_A77740((__int64)v24, v11, v12);
        }
        else
        {
          v14 = sub_A71AE0(&v27);
          sub_A77390((__int64)v24, v14);
        }
      }
    }
    v21 = sub_A7B2C0((__int64 *)&v21, v6, -1, (__int64)v24);
    if ( !a2 )
    {
      v15 = (__int64)&sub_24E54B0((unsigned __int8 *)a1)[-(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))] >> 5;
      if ( (_DWORD)v15 )
      {
        v16 = v15;
        for ( k = 0; k != v16; ++k )
        {
          v18 = sub_A744E0(&v22, k);
          sub_A74940((__int64)&v27, (__int64)v6, v18);
          v21 = sub_A7B2C0((__int64 *)&v21, v6, k + 6, (__int64)&v27);
          if ( v28 != &v29 )
            _libc_free((unsigned __int64)v28);
        }
      }
    }
    v3 = v21;
    if ( v25 != &v26 )
      _libc_free((unsigned __int64)v25);
  }
  return v3;
}
