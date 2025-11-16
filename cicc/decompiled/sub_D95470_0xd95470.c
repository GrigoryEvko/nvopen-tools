// Function: sub_D95470
// Address: 0xd95470
//
__int64 __fastcall sub_D95470(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 *v3; // rbx
  __int64 v4; // rax
  unsigned int v5; // r12d
  __int64 v7; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-58h]
  __int64 v9; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-48h]
  __int64 v11; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v12; // [rsp+28h] [rbp-38h]

  v2 = &a1[a2];
  v8 = 16;
  v7 = 1;
  if ( v2 == a1 )
    return (unsigned __int16)v7;
  v3 = a1;
  do
  {
    v4 = *v3;
    v10 = 16;
    v9 = *(unsigned __int16 *)(v4 + 26);
    sub_C49B30((__int64)&v11, (__int64)&v7, &v9);
    if ( v8 > 0x40 && v7 )
      j_j___libc_free_0_0(v7);
    v7 = v11;
    v8 = v12;
    if ( v10 > 0x40 && v9 )
      j_j___libc_free_0_0(v9);
    ++v3;
  }
  while ( v2 != v3 );
  if ( v8 <= 0x40 )
  {
    return (unsigned __int16)v7;
  }
  else
  {
    v5 = *(unsigned __int16 *)v7;
    j_j___libc_free_0_0(v7);
  }
  return v5;
}
