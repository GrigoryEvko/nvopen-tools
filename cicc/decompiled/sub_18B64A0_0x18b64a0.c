// Function: sub_18B64A0
// Address: 0x18b64a0
//
__int64 __fastcall sub_18B64A0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        char *a7,
        size_t a8)
{
  __int64 v9; // r14
  __int64 v10; // rdi
  __int64 v11; // rbx
  char v12; // al
  __int64 result; // rax
  __int64 *v14; // [rsp+0h] [rbp-60h] BYREF
  __int16 v15; // [rsp+10h] [rbp-50h]
  __int64 v16[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v17; // [rsp+30h] [rbp-30h] BYREF

  v9 = *a1;
  sub_18B61E0(v16, a2, a3, a4, a5, a6, a7, a8);
  v10 = a1[5];
  v15 = 260;
  v14 = v16;
  v11 = sub_15E57E0(v10, 0, 0, (__int64)&v14, a6, v9);
  if ( (__int64 *)v16[0] != &v17 )
    j_j___libc_free_0(v16[0], v17 + 1);
  v12 = *(_BYTE *)(v11 + 32) & 0xCF | 0x10;
  *(_BYTE *)(v11 + 32) = v12;
  result = v12 & 0xF;
  if ( (_BYTE)result != 9 )
    *(_BYTE *)(v11 + 33) |= 0x40u;
  return result;
}
