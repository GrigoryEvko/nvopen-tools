// Function: sub_2C167C0
// Address: 0x2c167c0
//
_QWORD *__fastcall sub_2C167C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r12
  const void *v8; // r8
  __int64 v9; // r14
  __int64 *v10; // r15
  __int64 v11; // r13
  __int64 v12; // r9
  _QWORD *v13; // r12
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 *v17; // rdi
  __int64 v18; // [rsp+8h] [rbp-88h]
  const void *v19; // [rsp+8h] [rbp-88h]
  __int64 v20; // [rsp+10h] [rbp-80h] BYREF
  __int64 v21; // [rsp+18h] [rbp-78h] BYREF
  __int64 *v22; // [rsp+20h] [rbp-70h] BYREF
  __int64 v23; // [rsp+28h] [rbp-68h]
  _BYTE dest[96]; // [rsp+30h] [rbp-60h] BYREF

  v7 = *(unsigned int *)(a1 + 56);
  v8 = *(const void **)(a1 + 48);
  v22 = (__int64 *)dest;
  v23 = 0x600000000LL;
  v9 = 8 * v7;
  if ( v7 > 6 )
  {
    v19 = v8;
    sub_C8D5F0((__int64)&v22, dest, v7, 8u, (__int64)v8, a6);
    v8 = v19;
    v17 = &v22[(unsigned int)v23];
  }
  else
  {
    v10 = (__int64 *)dest;
    if ( !v9 )
      goto LABEL_3;
    v17 = (__int64 *)dest;
  }
  memcpy(v17, v8, 8 * v7);
  v10 = v22;
  LODWORD(v9) = v23;
LABEL_3:
  v11 = *(_QWORD *)(a1 + 136);
  LODWORD(v23) = v9 + v7;
  v18 = (unsigned int)(v9 + v7);
  v13 = (_QWORD *)sub_22077B0(0x98u);
  if ( v13 )
  {
    v20 = *(_QWORD *)(v11 + 48);
    if ( v20 )
    {
      sub_2AAAFA0(&v20);
      v21 = v20;
      if ( v20 )
        sub_2AAAFA0(&v21);
    }
    else
    {
      v21 = 0;
    }
    sub_2AAF310((__int64)v13, 25, v10, v18, &v21, v12);
    sub_9C6650(&v21);
    sub_2BF0340((__int64)(v13 + 12), 1, v11, (__int64)v13, v14, v15);
    *v13 = &unk_4A231C8;
    v13[5] = &unk_4A23200;
    v13[12] = &unk_4A23238;
    sub_9C6650(&v20);
    *v13 = &unk_4A243A0;
    v13[5] = &unk_4A243E0;
    v13[12] = &unk_4A24418;
  }
  if ( v22 != (__int64 *)dest )
    _libc_free((unsigned __int64)v22);
  return v13;
}
