// Function: sub_96A450
// Address: 0x96a450
//
__int64 __fastcall sub_96A450(_QWORD *a1, double a2)
{
  char v2; // al
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v8; // r13
  __int64 v9; // r13
  _QWORD *i; // rbx
  __int64 v11; // [rsp+10h] [rbp-60h] BYREF
  _QWORD *v12; // [rsp+18h] [rbp-58h]
  _BYTE v13[64]; // [rsp+30h] [rbp-40h] BYREF

  v2 = *((_BYTE *)a1 + 8);
  if ( (v2 & 0xFD) != 0 )
  {
    if ( v2 != 3 )
      BUG();
    v8 = sub_C33320(a1);
    sub_C3B1B0(v13, a2);
    sub_C407B0(&v11, v13, v8);
    sub_C338F0(v13);
    v5 = sub_AC8EA0(*a1, &v11);
    v9 = sub_C33340();
    if ( v11 != v9 )
      goto LABEL_3;
    if ( !v12 )
      return v5;
    i = &v12[3 * *(v12 - 1)];
    while ( v12 != i )
    {
      i -= 3;
      if ( v9 == *i )
        sub_969EE0((__int64)i);
      else
        sub_C338F0(i);
    }
  }
  else
  {
    v3 = sub_C33320(a1);
    sub_C3B1B0(v13, a2);
    sub_C407B0(&v11, v13, v3);
    sub_C338F0(v13);
    v4 = sub_BCAC60(a1);
    sub_C41640(&v11, v4, 1, v13);
    v5 = sub_AC8EA0(*a1, &v11);
    v6 = sub_C33340();
    if ( v11 != v6 )
    {
LABEL_3:
      sub_C338F0(&v11);
      return v5;
    }
    if ( !v12 )
      return v5;
    for ( i = &v12[3 * *(v12 - 1)]; v12 != i; sub_969EE0((__int64)i) )
    {
      while ( 1 )
      {
        i -= 3;
        if ( v6 == *i )
          break;
        sub_C338F0(i);
        if ( v12 == i )
          goto LABEL_20;
      }
    }
  }
LABEL_20:
  j_j_j___libc_free_0_0(i - 1);
  return v5;
}
