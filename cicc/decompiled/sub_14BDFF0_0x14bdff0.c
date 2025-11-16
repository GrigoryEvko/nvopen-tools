// Function: sub_14BDFF0
// Address: 0x14bdff0
//
__int64 __fastcall sub_14BDFF0(
        __int64 a1,
        __int64 a2,
        unsigned __int8 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned int *v9; // rax
  unsigned int v10; // r15d
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // eax
  unsigned int v14; // r12d
  _QWORD *v16; // rax
  _QWORD *i; // rdx
  _QWORD v18[5]; // [rsp+10h] [rbp-A0h] BYREF
  _BYTE *v19; // [rsp+38h] [rbp-78h] BYREF
  __int64 v20; // [rsp+40h] [rbp-70h]
  _BYTE v21[48]; // [rsp+48h] [rbp-68h] BYREF
  int v22; // [rsp+78h] [rbp-38h]

  if ( !a6 || !*(_QWORD *)(a6 + 40) )
  {
    a6 = 0;
    if ( *(_BYTE *)(a1 + 16) > 0x17u )
    {
      a6 = *(_QWORD *)(a1 + 40);
      if ( a6 )
        a6 = a1;
    }
  }
  v18[0] = a2;
  v18[1] = a5;
  v18[3] = a7;
  v18[2] = a6;
  v18[4] = 0;
  v19 = v21;
  v20 = 0x600000000LL;
  v22 = 0;
  v9 = (unsigned int *)sub_16D40F0(qword_4FBB370);
  if ( v9 )
    v10 = *v9;
  else
    v10 = qword_4FBB370[2];
  v11 = (unsigned int)v20;
  v12 = v10;
  if ( v10 >= (unsigned __int64)(unsigned int)v20 )
  {
    if ( v10 <= (unsigned __int64)(unsigned int)v20 )
      goto LABEL_7;
    if ( v10 > (unsigned __int64)HIDWORD(v20) )
    {
      sub_16CD150(&v19, v21, v10, 8);
      v11 = (unsigned int)v20;
      v12 = v10;
    }
    v16 = &v19[8 * v11];
    for ( i = &v19[8 * v12]; i != v16; ++v16 )
    {
      if ( v16 )
        *v16 = 0;
    }
  }
  LODWORD(v20) = v10;
LABEL_7:
  LOBYTE(v13) = sub_14BDDF0(a1, a3, a4, (__int64)v18);
  v14 = v13;
  if ( v19 != v21 )
    _libc_free((unsigned __int64)v19);
  return v14;
}
