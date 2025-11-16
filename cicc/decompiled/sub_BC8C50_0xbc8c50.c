// Function: sub_BC8C50
// Address: 0xbc8c50
//
__int64 __fastcall sub_BC8C50(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v3; // r8
  unsigned int v5; // eax
  unsigned int *v6; // rdi
  unsigned int v7; // r12d
  unsigned int *v9; // [rsp+0h] [rbp-40h] BYREF
  __int64 v10; // [rsp+8h] [rbp-38h]
  _BYTE v11[48]; // [rsp+10h] [rbp-30h] BYREF

  v3 = 0;
  v9 = (unsigned int *)v11;
  v10 = 0x200000000LL;
  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
    v3 = sub_B91C10(a1, 2);
  v5 = sub_BC8BD0(v3, (__int64)&v9);
  v6 = v9;
  v7 = v5;
  if ( (_BYTE)v5 )
  {
    if ( (unsigned int)v10 > 2 )
    {
      v7 = 0;
    }
    else
    {
      *a2 = *v9;
      *a3 = v6[1];
    }
  }
  if ( v6 != (unsigned int *)v11 )
    _libc_free(v6, &v9);
  return v7;
}
