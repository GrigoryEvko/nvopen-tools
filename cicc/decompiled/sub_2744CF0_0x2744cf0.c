// Function: sub_2744CF0
// Address: 0x2744cf0
//
__int64 __fastcall sub_2744CF0(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned int v9; // r13d
  unsigned __int64 *v10; // rbx
  unsigned __int64 *v11; // r12
  unsigned __int64 v13[2]; // [rsp+0h] [rbp-170h] BYREF
  _BYTE v14[64]; // [rsp+10h] [rbp-160h] BYREF
  char *v15; // [rsp+50h] [rbp-120h] BYREF
  unsigned int v16; // [rsp+58h] [rbp-118h]
  char v17; // [rsp+60h] [rbp-110h] BYREF
  char *v18; // [rsp+A0h] [rbp-D0h]
  int v19; // [rsp+A8h] [rbp-C8h]
  char v20; // [rsp+B0h] [rbp-C0h] BYREF
  unsigned __int64 *v21; // [rsp+E0h] [rbp-90h]
  unsigned int v22; // [rsp+E8h] [rbp-88h]
  _BYTE v23[128]; // [rsp+F0h] [rbp-80h] BYREF

  v3 = a3;
  sub_2742E90(&v15, a3, 0x27u, a1, a2);
  if ( v16 > 1 && (v4 = (__int64)&v18[24 * v19], v4 == sub_27435A0((__int64)v18, v4, v3)) )
  {
    if ( v23[80] )
      v3 += 632;
    v13[1] = 0x800000000LL;
    v13[0] = (unsigned __int64)v14;
    if ( v16 )
      sub_27388F0((__int64)v13, (__int64)&v15, v5, v6, v7, v8);
    v9 = sub_30ADCF0(v3, v13);
    if ( (_BYTE *)v13[0] != v14 )
      _libc_free(v13[0]);
  }
  else
  {
    v9 = 0;
  }
  v10 = v21;
  v11 = &v21[10 * v22];
  if ( v21 != v11 )
  {
    do
    {
      v11 -= 10;
      if ( (unsigned __int64 *)*v11 != v11 + 2 )
        _libc_free(*v11);
    }
    while ( v10 != v11 );
    v11 = v21;
  }
  if ( v11 != (unsigned __int64 *)v23 )
    _libc_free((unsigned __int64)v11);
  if ( v18 != &v20 )
    _libc_free((unsigned __int64)v18);
  if ( v15 != &v17 )
    _libc_free((unsigned __int64)v15);
  return v9;
}
