// Function: sub_9718F0
// Address: 0x9718f0
//
__int64 __fastcall sub_9718F0(__int64 a1, __int64 a2, _BYTE *a3)
{
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v8; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v9; // [rsp+8h] [rbp-38h]
  __int64 v10; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-28h]

  v4 = sub_AE43F0(a3, *(_QWORD *)(a1 + 8));
  v9 = v4;
  if ( v4 > 0x40 )
  {
    sub_C43690(&v8, 0, 0);
    v5 = v8;
    v4 = v9;
  }
  else
  {
    v8 = 0;
    v5 = 0;
  }
  v10 = v5;
  v11 = v4;
  v9 = 0;
  v6 = sub_971820(a1, a2, (__int64)&v10, a3);
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  return v6;
}
