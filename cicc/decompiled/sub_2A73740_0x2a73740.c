// Function: sub_2A73740
// Address: 0x2a73740
//
__int64 *__fastcall sub_2A73740(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 *v4; // rdi
  __int64 *v5; // rax
  unsigned __int64 v6; // rdi
  __int64 *v7; // r12
  __int64 *v9; // rdi
  __int64 *v10; // rax
  __int64 *v11; // [rsp+0h] [rbp-30h] BYREF
  __int64 v12; // [rsp+8h] [rbp-28h]
  __int64 v13; // [rsp+10h] [rbp-20h] BYREF
  __int64 v14; // [rsp+18h] [rbp-18h]

  if ( a4 == 17 )
  {
    v9 = *(__int64 **)(a1 + 32);
    v13 = a2;
    v14 = a3;
    v11 = &v13;
    v12 = 0x200000002LL;
    v10 = sub_DC8BD0(v9, (__int64)&v11, 0, 0);
    v6 = (unsigned __int64)v11;
    v7 = v10;
    if ( v11 == &v13 )
      return v7;
    goto LABEL_5;
  }
  if ( a4 <= 0x11 )
  {
    if ( a4 == 13 )
    {
      v4 = *(__int64 **)(a1 + 32);
      v13 = a2;
      v14 = a3;
      v11 = &v13;
      v12 = 0x200000002LL;
      v5 = sub_DC7EB0(v4, (__int64)&v11, 0, 0);
      v6 = (unsigned __int64)v11;
      v7 = v5;
      if ( v11 == &v13 )
        return v7;
LABEL_5:
      _libc_free(v6);
      return v7;
    }
    if ( a4 == 15 )
      return sub_DCC810(*(__int64 **)(a1 + 32), a2, a3, 0, 0);
LABEL_13:
    BUG();
  }
  if ( a4 != 19 )
    goto LABEL_13;
  return (__int64 *)sub_DCB270(*(_QWORD *)(a1 + 32), a2, a3);
}
