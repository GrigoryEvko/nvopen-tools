// Function: sub_14EA060
// Address: 0x14ea060
//
__int64 __fastcall sub_14EA060(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned int a4)
{
  _BYTE *v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rdx
  bool v11; // zf
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  _BYTE *v15; // [rsp+0h] [rbp-80h] BYREF
  __int64 v16; // [rsp+8h] [rbp-78h]
  _BYTE s[112]; // [rsp+10h] [rbp-70h] BYREF

  v8 = s;
  v15 = s;
  v16 = 0x800000000LL;
  if ( a3 > 8 )
  {
    sub_16CD150(&v15, s, a3, 8);
    v8 = v15;
  }
  LODWORD(v16) = a3;
  if ( 8LL * (unsigned int)a3 )
  {
    memset(v8, 0, 8LL * (unsigned int)a3);
    v8 = v15;
  }
  v9 = 8 * a3;
  if ( 8 * a3 )
  {
    v10 = 0;
    do
    {
      while ( 1 )
      {
        v13 = *(_QWORD *)(a2 + v10);
        if ( (v13 & 1) != 0 )
          break;
        *(_QWORD *)&v8[v10] = v13 >> 1;
        v10 += 8;
        if ( v9 == v10 )
          goto LABEL_12;
      }
      v11 = v13 == 1;
      v12 = -(__int64)(v13 >> 1);
      if ( v11 )
        v12 = 0x8000000000000000LL;
      *(_QWORD *)&v8[v10] = v12;
      v10 += 8;
    }
    while ( v9 != v10 );
LABEL_12:
    v8 = v15;
  }
  sub_16A50F0(a1, a4, v8, (unsigned int)v16);
  if ( v15 != s )
    _libc_free((unsigned __int64)v15);
  return a1;
}
