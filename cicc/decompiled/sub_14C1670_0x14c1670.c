// Function: sub_14C1670
// Address: 0x14c1670
//
__int64 __fastcall sub_14C1670(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6, __int64 a7)
{
  unsigned int v7; // r12d
  unsigned int *v8; // rax
  unsigned int v9; // r15d
  __int64 v10; // rax
  unsigned int v11; // r15d
  _QWORD *v13; // rax
  _QWORD *i; // rdx
  __int64 v16; // [rsp+10h] [rbp-C0h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-B8h]
  __int64 v18; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v19; // [rsp+28h] [rbp-A8h]
  __int64 v20[5]; // [rsp+30h] [rbp-A0h] BYREF
  _BYTE *v21; // [rsp+58h] [rbp-78h] BYREF
  __int64 v22; // [rsp+60h] [rbp-70h]
  _BYTE v23[48]; // [rsp+68h] [rbp-68h] BYREF
  int v24; // [rsp+98h] [rbp-38h]

  v7 = a2;
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
  v20[0] = a3;
  v20[1] = a5;
  v20[3] = a7;
  v20[2] = a6;
  v20[4] = 0;
  v21 = v23;
  v22 = 0x600000000LL;
  v24 = 0;
  v8 = (unsigned int *)sub_16D40F0(qword_4FBB370);
  if ( v8 )
    v9 = *v8;
  else
    v9 = qword_4FBB370[2];
  v10 = (unsigned int)v22;
  if ( v9 >= (unsigned __int64)(unsigned int)v22 )
  {
    if ( v9 <= (unsigned __int64)(unsigned int)v22 )
      goto LABEL_7;
    if ( v9 > (unsigned __int64)HIDWORD(v22) )
    {
      sub_16CD150(&v21, v23, v9, 8);
      v10 = (unsigned int)v22;
    }
    v13 = &v21[8 * v10];
    for ( i = &v21[8 * v9]; i != v13; ++v13 )
    {
      if ( v13 )
        *v13 = 0;
    }
  }
  LODWORD(v22) = v9;
LABEL_7:
  v11 = *(_DWORD *)(a2 + 8);
  v17 = v11;
  if ( v11 > 0x40 )
  {
    sub_16A4EF0(&v16, 0, 0);
    v19 = v11;
    sub_16A4EF0(&v18, 0, 0);
  }
  else
  {
    v19 = v11;
    v16 = 0;
    v18 = 0;
  }
  sub_14B86A0((__int64 *)a1, (__int64)&v16, a4, v20);
  if ( *(_DWORD *)(a2 + 8) <= 0x40u )
    LOBYTE(v7) = (*(_QWORD *)a2 & ~v16) == 0;
  else
    v7 = sub_16A5A00(a2, &v16);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v21 != v23 )
    _libc_free((unsigned __int64)v21);
  return v7;
}
