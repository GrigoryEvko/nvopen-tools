// Function: sub_2B140F0
// Address: 0x2b140f0
//
__int64 __fastcall sub_2B140F0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r13
  _QWORD *v7; // rbx
  __int64 v8; // rdx
  _BYTE *v9; // r12
  __int64 *v10; // rsi
  __int64 v11; // r12
  __int64 *v13; // [rsp+10h] [rbp-70h] BYREF
  __int64 v14; // [rsp+18h] [rbp-68h]
  _BYTE v15[96]; // [rsp+20h] [rbp-60h] BYREF

  v6 = &a2[a3];
  v13 = (__int64 *)v15;
  v14 = 0x600000000LL;
  if ( v6 == a2 )
  {
    v8 = 0;
    v10 = (__int64 *)v15;
  }
  else
  {
    v7 = a2;
    v8 = 0;
    do
    {
      while ( 1 )
      {
        v9 = (_BYTE *)*v7;
        if ( *(_BYTE *)*v7 > 0x1Cu )
          break;
        if ( v6 == ++v7 )
          goto LABEL_8;
      }
      if ( v8 + 1 > (unsigned __int64)HIDWORD(v14) )
      {
        sub_C8D5F0((__int64)&v13, v15, v8 + 1, 8u, v8 + 1, a6);
        v8 = (unsigned int)v14;
      }
      ++v7;
      v13[v8] = (__int64)v9;
      v8 = (unsigned int)(v14 + 1);
      LODWORD(v14) = v14 + 1;
    }
    while ( v6 != v7 );
LABEL_8:
    v10 = v13;
  }
  v11 = sub_9B8FE0(a1, v10, v8);
  if ( v13 != (__int64 *)v15 )
    _libc_free((unsigned __int64)v13);
  return v11;
}
