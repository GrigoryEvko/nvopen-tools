// Function: sub_15942D0
// Address: 0x15942d0
//
__int64 __fastcall sub_15942D0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  int v4; // r15d
  _BYTE *v6; // rdi
  size_t v8; // rdx
  __int64 i; // rax
  __int64 v10; // r13
  __int64 v12; // [rsp+8h] [rbp-C8h]
  _BYTE *v13; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v14; // [rsp+18h] [rbp-B8h]
  _BYTE s[176]; // [rsp+20h] [rbp-B0h] BYREF

  v4 = a3;
  a3 = (unsigned int)a3;
  v6 = s;
  v13 = s;
  v14 = 0x1000000000LL;
  if ( (unsigned int)a3 > 0x10 )
  {
    v12 = (unsigned int)a3;
    sub_16CD150(&v13, s, (unsigned int)a3, 8);
    v6 = v13;
    a3 = v12;
  }
  v8 = 8 * a3;
  LODWORD(v14) = v4;
  if ( v8 )
  {
    memset(v6, 0, v8);
    v6 = v13;
  }
  if ( v4 )
  {
    for ( i = 0; ; i += 8 )
    {
      *(_QWORD *)&v6[i] = **(_QWORD **)(a2 + i);
      v6 = v13;
      if ( 8LL * (unsigned int)(v4 - 1) == i )
        break;
    }
  }
  v10 = sub_1645600(a1, v6, (unsigned int)v14, a4);
  if ( v13 != s )
    _libc_free((unsigned __int64)v13);
  return v10;
}
