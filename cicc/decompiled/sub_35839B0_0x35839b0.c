// Function: sub_35839B0
// Address: 0x35839b0
//
__int64 __fastcall sub_35839B0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  unsigned int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rsi
  _BYTE *v10; // rbx
  _BYTE *v11; // r12
  int v12; // r15d
  __int64 v13; // rsi
  __int64 v15; // [rsp+10h] [rbp-90h] BYREF
  int v16; // [rsp+18h] [rbp-88h]
  _BYTE *v17; // [rsp+20h] [rbp-80h] BYREF
  __int64 v18; // [rsp+28h] [rbp-78h]
  _BYTE v19[112]; // [rsp+30h] [rbp-70h] BYREF

  v6 = (__int64)(a2 + 40);
  v7 = 0;
  if ( (_QWORD *)(a2[40] & 0xFFFFFFFFFFFFFFF8LL) != a2 + 40 )
  {
    v8 = a2[4];
    v9 = a2[41];
    v16 = 0;
    v15 = v8;
    v17 = v19;
    v18 = 0x800000000LL;
    sub_3583570((__int64)&v17, v9, v6, a4, a5, a6);
    v10 = v17;
    v11 = &v17[8 * (unsigned int)v18];
    if ( v17 != v11 )
    {
      v12 = 0;
      do
      {
        v13 = *((_QWORD *)v11 - 1);
        v11 -= 8;
        v16 = v12++;
        v7 |= sub_3592050(&v15, v13);
      }
      while ( v10 != v11 );
      v11 = v17;
    }
    if ( v11 != v19 )
      _libc_free((unsigned __int64)v11);
  }
  return v7;
}
