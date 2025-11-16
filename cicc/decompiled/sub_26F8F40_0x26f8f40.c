// Function: sub_26F8F40
// Address: 0x26f8f40
//
__int64 __fastcall sub_26F8F40(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 *a4,
        __int64 a5,
        __int64 a6,
        unsigned __int8 *a7,
        size_t a8)
{
  __int64 v9; // r14
  _QWORD *v10; // rdi
  __int64 v11; // rbx
  char v12; // al
  __int64 result; // rax
  __int64 v14[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v15; // [rsp+10h] [rbp-60h] BYREF
  __int64 *v16; // [rsp+20h] [rbp-50h] BYREF
  __int16 v17; // [rsp+40h] [rbp-30h]

  v9 = *a1;
  sub_26F78E0(v14, a2, a3, a4, a5, a6, a7, a8);
  v10 = (_QWORD *)a1[7];
  v17 = 260;
  v16 = v14;
  v11 = sub_B30500(v10, 0, 0, (__int64)&v16, a6, v9);
  if ( (__int64 *)v14[0] != &v15 )
    j_j___libc_free_0(v14[0]);
  v12 = *(_BYTE *)(v11 + 32) & 0xCF | 0x10;
  *(_BYTE *)(v11 + 32) = v12;
  result = v12 & 0xF;
  if ( (_BYTE)result != 9 )
    *(_BYTE *)(v11 + 33) |= 0x40u;
  return result;
}
