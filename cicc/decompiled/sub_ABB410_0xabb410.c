// Function: sub_ABB410
// Address: 0xabb410
//
char __fastcall sub_ABB410(__int64 *a1, int a2, __int64 *a3)
{
  unsigned int v4; // r12d
  unsigned int v6; // eax
  int v7; // eax
  __int64 *v8; // r13
  __int64 *v9; // rsi
  int v10; // eax
  __int64 v11[2]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v12[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v13[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( sub_AAF7D0((__int64)a1) || (LOBYTE(v4) = sub_AAF7D0((__int64)a3), (_BYTE)v4) )
  {
    LOBYTE(v4) = 1;
  }
  else
  {
    switch ( a2 )
    {
      case ' ':
        v8 = sub_9876C0(a1);
        if ( !v8 )
          return v4;
        v9 = sub_9876C0(a3);
        if ( !v9 )
          return v4;
        return sub_AAD8B0((__int64)v8, v9);
      case '!':
        sub_ABB300((__int64)v12, (__int64)a1);
        LOBYTE(v4) = sub_AB1BB0((__int64)v12, (__int64)a3);
        sub_969240(v13);
        sub_969240(v12);
        return v4;
      case '"':
        sub_AB0A00((__int64)v11, (__int64)a1);
        sub_AB0910((__int64)v12, (__int64)a3);
        v10 = sub_C49970(v11, v12);
        goto LABEL_15;
      case '#':
        sub_AB0A00((__int64)v11, (__int64)a1);
        sub_AB0910((__int64)v12, (__int64)a3);
        v6 = ~(unsigned int)sub_C49970(v11, v12);
        goto LABEL_7;
      case '$':
        sub_AB0910((__int64)v11, (__int64)a1);
        sub_AB0A00((__int64)v12, (__int64)a3);
        v6 = sub_C49970(v11, v12);
        goto LABEL_7;
      case '%':
        sub_AB0910((__int64)v11, (__int64)a1);
        sub_AB0A00((__int64)v12, (__int64)a3);
        v7 = sub_C49970(v11, v12);
        goto LABEL_9;
      case '&':
        sub_AB14C0((__int64)v11, (__int64)a1);
        sub_AB13A0((__int64)v12, (__int64)a3);
        v10 = sub_C4C880(v11, v12);
LABEL_15:
        LOBYTE(v4) = v10 > 0;
        sub_969240(v12);
        sub_969240(v11);
        return v4;
      case '\'':
        sub_AB14C0((__int64)v11, (__int64)a1);
        sub_AB13A0((__int64)v12, (__int64)a3);
        v6 = ~(unsigned int)sub_C4C880(v11, v12);
        goto LABEL_7;
      case '(':
        sub_AB13A0((__int64)v11, (__int64)a1);
        sub_AB14C0((__int64)v12, (__int64)a3);
        v6 = sub_C4C880(v11, v12);
LABEL_7:
        v4 = v6 >> 31;
        sub_969240(v12);
        sub_969240(v11);
        break;
      case ')':
        sub_AB13A0((__int64)v11, (__int64)a1);
        sub_AB14C0((__int64)v12, (__int64)a3);
        v7 = sub_C4C880(v11, v12);
LABEL_9:
        LOBYTE(v4) = v7 <= 0;
        sub_969240(v12);
        sub_969240(v11);
        break;
      default:
        BUG();
    }
  }
  return v4;
}
