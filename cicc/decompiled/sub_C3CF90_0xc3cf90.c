// Function: sub_C3CF90
// Address: 0xc3cf90
//
void __fastcall sub_C3CF90(__int64 a1, char a2)
{
  _DWORD *v3; // rax
  _DWORD *v4; // rbx
  _QWORD *v5; // rdi
  _DWORD *v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // r14
  __int64 v9; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-58h]
  _QWORD v11[10]; // [rsp+20h] [rbp-50h] BYREF

  v10 = 64;
  v9 = 0x7FEFFFFFFFFFFFFFLL;
  v3 = sub_C33340();
  v4 = v3;
  if ( v3 == dword_3F657A0 )
  {
    sub_C3C640(v11, (__int64)v3, &v9);
    v5 = *(_QWORD **)(a1 + 8);
    v6 = (_DWORD *)v11[0];
    if ( (_DWORD *)*v5 != v4 )
      goto LABEL_3;
  }
  else
  {
    sub_C3B160((__int64)v11, dword_3F657A0, &v9);
    v5 = *(_QWORD **)(a1 + 8);
    v6 = (_DWORD *)v11[0];
    if ( (_DWORD *)*v5 != v4 )
    {
LABEL_3:
      if ( v6 != v4 )
      {
        sub_C33870((__int64)v5, (__int64)v11);
        goto LABEL_5;
      }
      if ( v5 == v11 )
        goto LABEL_39;
      sub_C338F0((__int64)v5);
      goto LABEL_24;
    }
  }
  if ( v6 != v4 )
  {
    if ( v5 == v11 )
      goto LABEL_6;
    sub_969EE0((__int64)v5);
LABEL_24:
    if ( v4 == (_DWORD *)v11[0] )
      sub_C3C840(v5, v11);
    else
      sub_C338E0((__int64)v5, (__int64)v11);
    goto LABEL_5;
  }
  if ( v5 == v11 )
    goto LABEL_39;
  sub_969EE0((__int64)v5);
  sub_C3C840(v5, v11);
LABEL_5:
  if ( (_DWORD *)v11[0] == v4 )
  {
LABEL_39:
    sub_969EE0((__int64)v11);
    goto LABEL_7;
  }
LABEL_6:
  sub_C338F0((__int64)v11);
LABEL_7:
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  v10 = 64;
  v9 = 0x7C8FFFFFFFFFFFFELL;
  if ( v4 == dword_3F657A0 )
    sub_C3C640(v11, (__int64)v4, &v9);
  else
    sub_C3B160((__int64)v11, dword_3F657A0, &v9);
  v7 = *(_QWORD *)(a1 + 8);
  v8 = (_QWORD *)(v7 + 24);
  if ( *(_DWORD **)(v7 + 24) != v4 )
  {
    if ( (_DWORD *)v11[0] != v4 )
    {
      sub_C33870(v7 + 24, (__int64)v11);
      goto LABEL_15;
    }
    if ( v8 == v11 )
      goto LABEL_33;
    sub_C338F0(v7 + 24);
    goto LABEL_28;
  }
  if ( (_DWORD *)v11[0] != v4 )
  {
    if ( v8 == v11 )
      goto LABEL_16;
    sub_969EE0(v7 + 24);
LABEL_28:
    if ( v4 == (_DWORD *)v11[0] )
      sub_C3C840(v8, v11);
    else
      sub_C338E0((__int64)v8, (__int64)v11);
    goto LABEL_15;
  }
  if ( v8 == v11 )
    goto LABEL_33;
  sub_969EE0(v7 + 24);
  sub_C3C840(v8, v11);
LABEL_15:
  if ( (_DWORD *)v11[0] == v4 )
  {
LABEL_33:
    sub_969EE0((__int64)v11);
    goto LABEL_17;
  }
LABEL_16:
  sub_C338F0((__int64)v11);
LABEL_17:
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  if ( a2 )
    sub_C3CCB0(a1);
}
