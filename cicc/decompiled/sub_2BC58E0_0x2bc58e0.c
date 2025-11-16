// Function: sub_2BC58E0
// Address: 0x2bc58e0
//
void __fastcall sub_2BC58E0(__int64 a1, __int64 (__fastcall *a2)(__int64, __int64, __int64), __int64 a3)
{
  __int64 *v4; // r14
  __int64 v5; // rax
  __int64 *v6; // r10
  __int64 v7; // rcx
  char *v8; // rax
  unsigned __int64 v9; // r12
  __int64 *v10; // [rsp+0h] [rbp-40h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v4 = *(__int64 **)a1;
  v5 = 8LL * *(unsigned int *)(a1 + 8);
  v6 = (__int64 *)(*(_QWORD *)a1 + v5);
  v7 = v5 >> 3;
  if ( v5 )
  {
    while ( 1 )
    {
      v10 = v6;
      v11 = v7;
      v8 = (char *)sub_2207800(8 * v7);
      v6 = v10;
      v9 = (unsigned __int64)v8;
      if ( v8 )
        break;
      v7 = v11 >> 1;
      if ( !(v11 >> 1) )
        goto LABEL_5;
    }
    sub_2BC57C0(v4, v10, v8, (void *)v11, a2, a3);
    j_j___libc_free_0(v9);
  }
  else
  {
LABEL_5:
    sub_2BB6BA0(v4, v6, a2, a3);
    j_j___libc_free_0(0);
  }
}
