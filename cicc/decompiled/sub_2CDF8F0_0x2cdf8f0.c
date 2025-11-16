// Function: sub_2CDF8F0
// Address: 0x2cdf8f0
//
_QWORD *__fastcall sub_2CDF8F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // r13
  _QWORD *result; // rax
  __int64 v7; // r12
  __int64 v8; // [rsp-10h] [rbp-90h]
  unsigned __int64 v9[2]; // [rsp+0h] [rbp-80h] BYREF
  _BYTE v10[16]; // [rsp+10h] [rbp-70h] BYREF
  char *v11[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v12; // [rsp+30h] [rbp-50h] BYREF
  __int16 v13; // [rsp+40h] [rbp-40h]

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 72LL);
  if ( (unsigned __int8)sub_CE9220(v2) )
  {
    v10[0] = 0;
    v9[0] = (unsigned __int64)v10;
    v9[1] = 0;
    sub_B2BE50(v2);
    sub_2C75F20((__int64)v11, (__int64 *)(a1 + 48));
    sub_2241490(v9, v11[0], (size_t)v11[1]);
    if ( (__int64 *)v11[0] != &v12 )
      j_j___libc_free_0((unsigned __int64)v11[0]);
    sub_2241490(v9, *(char **)a2, *(_QWORD *)(a2 + 8));
    sub_CEB650(v9);
    if ( (_BYTE *)v9[0] != v10 )
      j_j___libc_free_0(v9[0]);
  }
  v3 = 0;
  v4 = sub_B6E160(*(__int64 **)(v2 + 40), 0x162u, 0, 0);
  v13 = 257;
  v5 = v4;
  if ( v4 )
    v3 = *(_QWORD *)(v4 + 24);
  result = sub_BD2C40(88, 1u);
  v7 = (__int64)result;
  if ( result )
  {
    sub_B44260((__int64)result, **(_QWORD **)(v3 + 16), 56, 1u, a1 + 24, 0);
    *(_QWORD *)(v7 + 72) = 0;
    sub_B4A290(v7, v3, v5, 0, 0, (__int64)v11, 0, 0);
    return (_QWORD *)v8;
  }
  return result;
}
