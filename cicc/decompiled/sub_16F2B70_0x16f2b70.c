// Function: sub_16F2B70
// Address: 0x16f2b70
//
__int64 *__fastcall sub_16F2B70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 *result; // rax
  __int64 **v6; // rbx
  __int64 **v7; // r13
  unsigned __int8 **v8; // r15
  unsigned __int8 **v9; // r14
  __int64 *v10; // rax
  __int64 *v11; // r12
  __int64 *v12; // r9
  __int64 *v13; // [rsp+0h] [rbp-A0h]
  __int64 v14; // [rsp+8h] [rbp-98h]
  unsigned __int8 **v15[4]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v16[2]; // [rsp+50h] [rbp-50h] BYREF
  _QWORD v17[8]; // [rsp+60h] [rbp-40h] BYREF

  *(_QWORD *)(a1 + 16) = 0;
  result = (__int64 *)sub_16F23B0((unsigned __int8 *)0xFFFFFFFFFFFFFFFFLL, 0, 0, a4, a5);
  if ( !(_BYTE)result )
  {
    sub_16F2420(v16, (unsigned __int8 *)0xFFFFFFFFFFFFFFFFLL, 0);
    sub_16F25B0(v15, (__int64)v16);
    v9 = v15[0];
    v8 = v15[2];
    v14 = (__int64)v15[1];
    result = v17;
    if ( (_QWORD *)v16[0] != v17 )
      result = (__int64 *)j_j___libc_free_0(v16[0], v17[0] + 1LL);
    v6 = *(__int64 ***)(a1 + 8);
    v7 = &v6[8 * (unsigned __int64)*(unsigned int *)(a1 + 24)];
    if ( v7 == v6 )
    {
LABEL_15:
      if ( v9 )
      {
        if ( *v9 != (unsigned __int8 *)(v9 + 2) )
          j_j___libc_free_0(*v9, v9[2] + 1);
        return (__int64 *)j_j___libc_free_0(v9, 32);
      }
      return result;
    }
    while ( 1 )
    {
LABEL_12:
      while ( !v6 )
      {
LABEL_11:
        v6 += 8;
        if ( v6 == v7 )
          goto LABEL_15;
      }
      *v6 = 0;
      v6[1] = 0;
      v6[2] = 0;
      if ( v9 )
      {
        v10 = (__int64 *)sub_22077B0(32);
        v11 = v10;
        if ( v10 )
        {
          *v10 = (__int64)(v10 + 2);
          sub_16F1520(v10, *v9, (__int64)&v9[1][(_QWORD)*v9]);
        }
        v12 = *v6;
        *v6 = v11;
        if ( v12 )
        {
          if ( (__int64 *)*v12 != v12 + 2 )
          {
            v13 = v12;
            j_j___libc_free_0(*v12, v12[2] + 1);
            v12 = v13;
          }
          j_j___libc_free_0(v12, 32);
          v11 = *v6;
        }
        result = (__int64 *)v11[1];
        v6[1] = (__int64 *)*v11;
        v6[2] = result;
        goto LABEL_11;
      }
      result = (__int64 *)v14;
      v6[2] = (__int64 *)v8;
      v6 += 8;
      *(v6 - 7) = (__int64 *)v14;
      if ( v6 == v7 )
        goto LABEL_15;
    }
  }
  v6 = *(__int64 ***)(a1 + 8);
  v7 = &v6[8 * (unsigned __int64)*(unsigned int *)(a1 + 24)];
  if ( v7 != v6 )
  {
    v14 = -1;
    v8 = 0;
    v9 = 0;
    goto LABEL_12;
  }
  return result;
}
