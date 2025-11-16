// Function: sub_14BB090
// Address: 0x14bb090
//
void __fastcall sub_14BB090(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  unsigned int *v9; // rax
  unsigned int v10; // r15d
  __int64 v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *i; // rdx
  __int64 v15[5]; // [rsp+10h] [rbp-A0h] BYREF
  _BYTE *v16; // [rsp+38h] [rbp-78h] BYREF
  __int64 v17; // [rsp+40h] [rbp-70h]
  _BYTE v18[48]; // [rsp+48h] [rbp-68h] BYREF
  int v19; // [rsp+78h] [rbp-38h]

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
  v15[0] = a3;
  v15[1] = a5;
  v15[3] = a7;
  v15[2] = a6;
  v15[4] = a8;
  v16 = v18;
  v17 = 0x600000000LL;
  v19 = 0;
  v9 = (unsigned int *)sub_16D40F0(qword_4FBB370);
  if ( v9 )
    v10 = *v9;
  else
    v10 = qword_4FBB370[2];
  v11 = (unsigned int)v17;
  v12 = v10;
  if ( v10 >= (unsigned __int64)(unsigned int)v17 )
  {
    if ( v10 <= (unsigned __int64)(unsigned int)v17 )
      goto LABEL_7;
    if ( v10 > (unsigned __int64)HIDWORD(v17) )
    {
      sub_16CD150(&v16, v18, v10, 8);
      v11 = (unsigned int)v17;
      v12 = v10;
    }
    v13 = &v16[8 * v11];
    for ( i = &v16[8 * v12]; i != v13; ++v13 )
    {
      if ( v13 )
        *v13 = 0;
    }
  }
  LODWORD(v17) = v10;
LABEL_7:
  sub_14B86A0((__int64 *)a1, a2, a4, v15);
  if ( v16 != v18 )
    _libc_free((unsigned __int64)v16);
}
