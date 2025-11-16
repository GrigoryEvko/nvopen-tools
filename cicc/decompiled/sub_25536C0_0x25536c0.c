// Function: sub_25536C0
// Address: 0x25536c0
//
unsigned __int8 *__fastcall sub_25536C0(__int64 a1, __int64 *a2, __int64 a3, char a4)
{
  unsigned __int8 *v6; // r12
  __int64 v7; // rcx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-38h]

  v11 = sub_AE43F0(a3, *(_QWORD *)(a1 + 8));
  if ( v11 > 0x40 )
    sub_C43690((__int64)&v10, 0, 0);
  else
    v10 = 0;
  v6 = sub_BD45C0((unsigned __int8 *)a1, a3, (__int64)&v10, a4, 0, 0, 0, 0);
  if ( v11 > 0x40 )
  {
    v9 = v10;
    *a2 = *(_QWORD *)v10;
    j_j___libc_free_0_0(v9);
  }
  else
  {
    v7 = 0;
    if ( v11 )
      v7 = (__int64)(v10 << (64 - (unsigned __int8)v11)) >> (64 - (unsigned __int8)v11);
    *a2 = v7;
  }
  return v6;
}
