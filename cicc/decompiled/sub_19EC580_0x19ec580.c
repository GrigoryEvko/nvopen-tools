// Function: sub_19EC580
// Address: 0x19ec580
//
__int64 __fastcall sub_19EC580(_BYTE *a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  __int64 v6; // rbx
  unsigned int v7; // r12d
  _BYTE *v9[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v10[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v11[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v12[8]; // [rsp+40h] [rbp-40h] BYREF

  v6 = sub_16BAF20();
  if ( a3 )
  {
    v11[0] = (__int64)v12;
    sub_19E1620(v11, a3, (__int64)&a3[a4]);
    if ( a1 )
    {
LABEL_3:
      v9[0] = v10;
      sub_19E1620((__int64 *)v9, a1, (__int64)&a1[a2]);
      goto LABEL_4;
    }
  }
  else
  {
    v11[1] = 0;
    v11[0] = (__int64)v12;
    LOBYTE(v12[0]) = 0;
    if ( a1 )
      goto LABEL_3;
  }
  v9[1] = 0;
  v9[0] = v10;
  LOBYTE(v10[0]) = 0;
LABEL_4:
  v7 = sub_14C9E50(v6, v9, (__int64)v11);
  if ( (_QWORD *)v9[0] != v10 )
    j_j___libc_free_0(v9[0], v10[0] + 1LL);
  if ( (_QWORD *)v11[0] != v12 )
    j_j___libc_free_0(v11[0], v12[0] + 1LL);
  return v7;
}
