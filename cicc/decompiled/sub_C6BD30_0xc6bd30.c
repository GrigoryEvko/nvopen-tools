// Function: sub_C6BD30
// Address: 0xc6bd30
//
__int64 *__fastcall sub_C6BD30(__int64 a1)
{
  __int64 *result; // rax
  __int64 **v2; // rbx
  __int64 **v3; // r13
  __int64 *v4; // r15
  __int64 *v5; // r14
  __int64 *v6; // rax
  __int64 *v7; // r12
  __int64 *v8; // r9
  __int64 *v9; // [rsp+0h] [rbp-A0h]
  __int64 v10; // [rsp+8h] [rbp-98h]
  __int64 *v11[4]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v12[2]; // [rsp+50h] [rbp-50h] BYREF
  _QWORD v13[8]; // [rsp+60h] [rbp-40h] BYREF

  *(_QWORD *)(a1 + 16) = 0;
  result = (__int64 *)sub_C6A630((char *)0xFFFFFFFFFFFFFFFFLL, 0, 0);
  if ( !(_BYTE)result )
  {
    sub_C6B0E0(v12, -1, 0);
    sub_C6B270(v11, (__int64)v12);
    v5 = v11[0];
    v4 = v11[2];
    v10 = (__int64)v11[1];
    result = v13;
    if ( (_QWORD *)v12[0] != v13 )
      result = (__int64 *)j_j___libc_free_0(v12[0], v13[0] + 1LL);
    v2 = *(__int64 ***)(a1 + 8);
    v3 = &v2[8 * (unsigned __int64)*(unsigned int *)(a1 + 24)];
    if ( v3 == v2 )
    {
LABEL_15:
      if ( v5 )
      {
        if ( (__int64 *)*v5 != v5 + 2 )
          j_j___libc_free_0(*v5, v5[2] + 1);
        return (__int64 *)j_j___libc_free_0(v5, 32);
      }
      return result;
    }
    while ( 1 )
    {
LABEL_12:
      while ( !v2 )
      {
LABEL_11:
        v2 += 8;
        if ( v2 == v3 )
          goto LABEL_15;
      }
      *v2 = 0;
      v2[1] = 0;
      v2[2] = 0;
      if ( v5 )
      {
        v6 = (__int64 *)sub_22077B0(32);
        v7 = v6;
        if ( v6 )
        {
          *v6 = (__int64)(v6 + 2);
          sub_C68E20(v6, (_BYTE *)*v5, *v5 + v5[1]);
        }
        v8 = *v2;
        *v2 = v7;
        if ( v8 )
        {
          if ( (__int64 *)*v8 != v8 + 2 )
          {
            v9 = v8;
            j_j___libc_free_0(*v8, v8[2] + 1);
            v8 = v9;
          }
          j_j___libc_free_0(v8, 32);
          v7 = *v2;
        }
        result = (__int64 *)v7[1];
        v2[1] = (__int64 *)*v7;
        v2[2] = result;
        goto LABEL_11;
      }
      result = (__int64 *)v10;
      v2[2] = v4;
      v2 += 8;
      *(v2 - 7) = (__int64 *)v10;
      if ( v2 == v3 )
        goto LABEL_15;
    }
  }
  v2 = *(__int64 ***)(a1 + 8);
  v3 = &v2[8 * (unsigned __int64)*(unsigned int *)(a1 + 24)];
  if ( v3 != v2 )
  {
    v10 = -1;
    v4 = 0;
    v5 = 0;
    goto LABEL_12;
  }
  return result;
}
