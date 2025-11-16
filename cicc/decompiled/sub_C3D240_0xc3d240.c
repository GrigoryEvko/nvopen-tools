// Function: sub_C3D240
// Address: 0xc3d240
//
__int64 __fastcall sub_C3D240(__int64 a1, char a2)
{
  _DWORD *v3; // rax
  _DWORD *v4; // r12
  _QWORD *v5; // rdi
  _DWORD *v6; // rax
  void **v7; // rdi
  unsigned __int8 *v9; // rdi
  _QWORD *i; // rbx
  __int64 v11; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-58h]
  _DWORD *v13; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v14; // [rsp+28h] [rbp-48h]

  v12 = 64;
  v11 = 0x360000000000000LL;
  v3 = sub_C33340();
  v4 = v3;
  if ( v3 == dword_3F657A0 )
  {
    sub_C3C640(&v13, (__int64)v3, &v11);
    v5 = *(_QWORD **)(a1 + 8);
    v6 = v13;
    if ( (_DWORD *)*v5 != v4 )
      goto LABEL_3;
  }
  else
  {
    sub_C3B160((__int64)&v13, dword_3F657A0, &v11);
    v5 = *(_QWORD **)(a1 + 8);
    v6 = v13;
    if ( (_DWORD *)*v5 != v4 )
    {
LABEL_3:
      if ( v4 != v6 )
      {
        sub_C33870((__int64)v5, (__int64)&v13);
        goto LABEL_5;
      }
      if ( v5 == &v13 )
        goto LABEL_25;
      sub_C338F0((__int64)v5);
      goto LABEL_15;
    }
  }
  if ( v4 != v6 )
  {
    if ( v5 == &v13 )
      goto LABEL_6;
    sub_969EE0((__int64)v5);
LABEL_15:
    if ( v4 == v13 )
      sub_C3C840(v5, &v13);
    else
      sub_C338E0((__int64)v5, (__int64)&v13);
    goto LABEL_5;
  }
  if ( v5 == &v13 )
    goto LABEL_25;
  sub_969EE0((__int64)v5);
  sub_C3C840(v5, &v13);
LABEL_5:
  if ( v13 != v4 )
  {
LABEL_6:
    sub_C338F0((__int64)&v13);
    goto LABEL_7;
  }
LABEL_25:
  if ( v14 )
  {
    for ( i = &v14[3 * *(v14 - 1)]; v14 != i; sub_969EE0((__int64)i) )
    {
      while ( 1 )
      {
        i -= 3;
        if ( v4 == (_DWORD *)*i )
          break;
        sub_C338F0((__int64)i);
        if ( v14 == i )
          goto LABEL_27;
      }
    }
LABEL_27:
    j_j_j___libc_free_0_0(i - 1);
  }
LABEL_7:
  if ( v12 > 0x40 )
  {
    if ( v11 )
      j_j___libc_free_0_0(v11);
  }
  if ( a2 )
  {
    v9 = *(unsigned __int8 **)(a1 + 8);
    if ( v4 != *(_DWORD **)v9 )
    {
      sub_C34440(v9);
      v7 = (void **)(*(_QWORD *)(a1 + 8) + 24LL);
      if ( v4 != *v7 )
        return sub_C37310((__int64)v7, 0);
      return sub_C3CEB0(v7, 0);
    }
    sub_C3CCB0((__int64)v9);
  }
  v7 = (void **)(*(_QWORD *)(a1 + 8) + 24LL);
  if ( v4 != *v7 )
    return sub_C37310((__int64)v7, 0);
  return sub_C3CEB0(v7, 0);
}
