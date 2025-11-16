// Function: sub_9B0030
// Address: 0x9b0030
//
__int64 __fastcall sub_9B0030(__int64 a1, __m128i *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  unsigned int v4; // r12d
  unsigned __int64 v6; // [rsp+0h] [rbp-70h] BYREF
  __int64 v7; // [rsp+8h] [rbp-68h]
  unsigned int v8; // [rsp+10h] [rbp-60h]
  __int64 v9; // [rsp+18h] [rbp-58h]
  unsigned int v10; // [rsp+20h] [rbp-50h]
  unsigned __int64 v11; // [rsp+30h] [rbp-40h] BYREF
  __int64 v12; // [rsp+38h] [rbp-38h]
  unsigned int v13; // [rsp+40h] [rbp-30h]
  __int64 v14; // [rsp+48h] [rbp-28h]
  unsigned int v15; // [rsp+50h] [rbp-20h]

  v2 = *(_QWORD *)(a1 - 32);
  v13 = 1;
  v12 = 0;
  v15 = 1;
  v11 = v2 & 0xFFFFFFFFFFFFFFFBLL;
  v3 = *(_QWORD *)(a1 - 64);
  v14 = 0;
  v8 = 1;
  v6 = v3 & 0xFFFFFFFFFFFFFFFBLL;
  v7 = 0;
  v10 = 1;
  v9 = 0;
  v4 = sub_9AFD60((__int64 *)&v6, (__int64 *)&v11, a1, a2);
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  if ( v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  return v4;
}
