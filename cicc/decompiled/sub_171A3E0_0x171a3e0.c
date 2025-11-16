// Function: sub_171A3E0
// Address: 0x171a3e0
//
__int64 __fastcall sub_171A3E0(__int64 a1, __int64 a2, int a3, double a4, double a5, double a6)
{
  _QWORD *v6; // r13
  void *v9; // rax
  void *v10; // r14
  _QWORD *v11; // rdi
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rsi
  __int64 v16; // rbx
  void *v17; // [rsp+8h] [rbp-48h] BYREF
  __int64 v18; // [rsp+10h] [rbp-40h]

  v6 = (_QWORD *)(a1 + 8);
  v9 = sub_16982C0();
  v10 = v9;
  if ( a3 < 0 )
  {
    v13 = -a3;
    if ( (void *)a2 == v9 )
      sub_169C630(&v17, (__int64)v9, v13);
    else
      sub_1699170((__int64)&v17, a2, v13);
    if ( v17 == v10 )
      sub_169C8D0((__int64)&v17, a4, a5, a6);
    else
      sub_1699490((__int64)&v17);
    if ( v17 == v10 )
      sub_169C7E0(v6, &v17);
    else
      sub_1698450((__int64)v6, (__int64)&v17);
    if ( v17 == v10 )
    {
      v14 = v18;
      if ( v18 )
      {
        v15 = 32LL * *(_QWORD *)(v18 - 8);
        v16 = v18 + v15;
        if ( v18 != v18 + v15 )
        {
          do
          {
            v16 -= 32;
            sub_127D120((_QWORD *)(v16 + 8));
          }
          while ( v14 != v16 );
        }
        j_j_j___libc_free_0_0(v14 - 8);
      }
    }
    else
    {
      sub_1698460((__int64)&v17);
    }
  }
  else
  {
    v11 = (_QWORD *)(a1 + 8);
    if ( (void *)a2 == v9 )
      sub_169C630(v11, a2, a3);
    else
      sub_1699170((__int64)v11, a2, a3);
  }
  return a1;
}
