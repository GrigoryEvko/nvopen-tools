// Function: sub_2743410
// Address: 0x2743410
//
__int64 __fastcall sub_2743410(__int64 a1, unsigned int a2, __int64 a3, _BYTE *a4)
{
  unsigned int v4; // r13d
  unsigned __int64 *v5; // rbx
  unsigned __int64 *v6; // r12
  char *v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r12
  unsigned __int64 v14[2]; // [rsp+0h] [rbp-170h] BYREF
  _BYTE v15[64]; // [rsp+10h] [rbp-160h] BYREF
  char *v16; // [rsp+50h] [rbp-120h] BYREF
  int v17; // [rsp+58h] [rbp-118h]
  char v18; // [rsp+60h] [rbp-110h] BYREF
  char *v19; // [rsp+A0h] [rbp-D0h]
  int v20; // [rsp+A8h] [rbp-C8h]
  char v21; // [rsp+B0h] [rbp-C0h] BYREF
  unsigned __int64 *v22; // [rsp+E0h] [rbp-90h]
  unsigned int v23; // [rsp+E8h] [rbp-88h]
  _BYTE v24[128]; // [rsp+F0h] [rbp-80h] BYREF

  sub_2742E90(&v16, a1, a2, a3, a4);
  if ( v17 && (v8 = &v19[24 * v20], v8 == (char *)sub_27435A0(v19, v8, a1)) )
  {
    v13 = a1;
    if ( v24[80] )
      v13 = a1 + 632;
    v14[1] = 0x800000000LL;
    v14[0] = (unsigned __int64)v15;
    if ( v17 )
      sub_27388F0((__int64)v14, (__int64)&v16, v9, v10, v11, v12);
    v4 = sub_30ADCF0(v13, v14);
    if ( (_BYTE *)v14[0] != v15 )
      _libc_free(v14[0]);
  }
  else
  {
    v4 = 0;
  }
  v5 = v22;
  v6 = &v22[10 * v23];
  if ( v22 != v6 )
  {
    do
    {
      v6 -= 10;
      if ( (unsigned __int64 *)*v6 != v6 + 2 )
        _libc_free(*v6);
    }
    while ( v5 != v6 );
    v6 = v22;
  }
  if ( v6 != (unsigned __int64 *)v24 )
    _libc_free((unsigned __int64)v6);
  if ( v19 != &v21 )
    _libc_free((unsigned __int64)v19);
  if ( v16 != &v18 )
    _libc_free((unsigned __int64)v16);
  return v4;
}
