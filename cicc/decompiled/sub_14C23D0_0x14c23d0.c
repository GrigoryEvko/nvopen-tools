// Function: sub_14C23D0
// Address: 0x14c23d0
//
__int64 __fastcall sub_14C23D0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int *v7; // rax
  unsigned int v8; // r14d
  __int64 v9; // rax
  unsigned int v10; // r12d
  _QWORD *v12; // rax
  _QWORD *i; // rdx
  __int64 v14[5]; // [rsp+0h] [rbp-A0h] BYREF
  _BYTE *v15; // [rsp+28h] [rbp-78h] BYREF
  __int64 v16; // [rsp+30h] [rbp-70h]
  _BYTE v17[48]; // [rsp+38h] [rbp-68h] BYREF
  int v18; // [rsp+68h] [rbp-38h]

  if ( !a5 || !*(_QWORD *)(a5 + 40) )
  {
    a5 = 0;
    if ( *(_BYTE *)(a1 + 16) > 0x17u )
    {
      a5 = *(_QWORD *)(a1 + 40);
      if ( a5 )
        a5 = a1;
    }
  }
  v14[0] = a2;
  v14[1] = a4;
  v14[2] = a5;
  v14[3] = a6;
  v14[4] = 0;
  v15 = v17;
  v16 = 0x600000000LL;
  v18 = 0;
  v7 = (unsigned int *)sub_16D40F0(qword_4FBB370);
  if ( v7 )
    v8 = *v7;
  else
    v8 = qword_4FBB370[2];
  v9 = (unsigned int)v16;
  if ( v8 >= (unsigned __int64)(unsigned int)v16 )
  {
    if ( v8 <= (unsigned __int64)(unsigned int)v16 )
      goto LABEL_7;
    if ( v8 > (unsigned __int64)HIDWORD(v16) )
    {
      sub_16CD150(&v15, v17, v8, 8);
      v9 = (unsigned int)v16;
    }
    v12 = &v15[8 * v9];
    for ( i = &v15[8 * v8]; i != v12; ++v12 )
    {
      if ( v12 )
        *v12 = 0;
    }
  }
  LODWORD(v16) = v8;
LABEL_7:
  v10 = sub_14C18C0((__int64 *)a1, a3, v14);
  if ( v15 != v17 )
    _libc_free((unsigned __int64)v15);
  return v10;
}
