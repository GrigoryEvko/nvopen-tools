// Function: sub_B8BAB0
// Address: 0xb8bab0
//
__int64 *__fastcall sub_B8BAB0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r12
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 *result; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // [rsp+0h] [rbp-60h]
  __int64 v13; // [rsp+8h] [rbp-58h] BYREF
  __int64 *v14; // [rsp+10h] [rbp-50h]
  __int64 v15; // [rsp+18h] [rbp-48h]
  __int64 v16[8]; // [rsp+20h] [rbp-40h] BYREF

  v13 = a2;
  v4 = *(_QWORD *)sub_B8B720(a1 + 568, &v13);
  if ( v4 )
  {
    v5 = v4 + 568;
  }
  else
  {
    v9 = sub_22077B0(1296);
    v10 = v9;
    if ( v9 )
    {
      v12 = v9;
      sub_B844F0(v9);
      v10 = v12;
      v11 = v12 + 568;
      v5 = v12 + 568;
    }
    else
    {
      v5 = 568;
      v11 = 0;
    }
    *(_QWORD *)(v10 + 184) = v11;
    *(_QWORD *)sub_B8B720(a1 + 568, &v13) = v10;
  }
  v6 = sub_B85AD0(*(_QWORD *)(a1 + 184), a3[2]);
  if ( v6 && *(_BYTE *)(v6 + 41) && (v8 = sub_B811E0(v5, a3[2])) != 0 )
    a3 = (__int64 *)v8;
  else
    sub_B8B080(v5, a3);
  v14 = v16;
  v16[0] = (__int64)a3;
  v15 = 0x100000001LL;
  result = sub_B87B60(v5, v16, 1, v13);
  if ( v14 != v16 )
    return (__int64 *)_libc_free(v14, v16);
  return result;
}
