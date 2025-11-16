// Function: sub_2A381E0
// Address: 0x2a381e0
//
void __fastcall sub_2A381E0(_BYTE *a1, char a2, char a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12[2]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v13[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v14[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v15[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v16; // [rsp+50h] [rbp-40h]
  __int64 v17; // [rsp+58h] [rbp-38h]

  if ( !a1 || !*a1 )
    goto LABEL_3;
  sub_B18290(a4, " Inlined: ", 0xAu);
  v12[0] = (__int64)v13;
  sub_2A37520(v12, "StoreInlined", (__int64)"");
  v14[0] = (__int64)v15;
  sub_2A37520(v14, "true", (__int64)"");
  v16 = 0;
  v17 = 0;
  v6 = sub_2A38130(a4, (__int64)v12);
  sub_B18290(v6, ".", 1u);
  if ( (_QWORD *)v14[0] != v15 )
    j_j___libc_free_0(v14[0]);
  if ( (_QWORD *)v12[0] != v13 )
  {
    j_j___libc_free_0(v12[0]);
    if ( !a2 )
      goto LABEL_4;
  }
  else
  {
LABEL_3:
    if ( !a2 )
      goto LABEL_4;
  }
  sub_B18290(a4, " Volatile: ", 0xBu);
  v12[0] = (__int64)v13;
  sub_2A37520(v12, "StoreVolatile", (__int64)"");
  v14[0] = (__int64)v15;
  sub_2A37520(v14, "true", (__int64)"");
  v16 = 0;
  v17 = 0;
  v7 = sub_2A38130(a4, (__int64)v12);
  sub_B18290(v7, ".", 1u);
  if ( (_QWORD *)v14[0] != v15 )
    j_j___libc_free_0(v14[0]);
  if ( (_QWORD *)v12[0] == v13 )
  {
LABEL_4:
    if ( a3 )
      goto LABEL_18;
LABEL_5:
    if ( a1 )
      goto LABEL_6;
LABEL_39:
    sub_B17B50(a4);
    goto LABEL_7;
  }
  j_j___libc_free_0(v12[0]);
  if ( !a3 )
    goto LABEL_5;
LABEL_18:
  sub_B18290(a4, " Atomic: ", 9u);
  v12[0] = (__int64)v13;
  sub_2A37520(v12, "StoreAtomic", (__int64)"");
  v14[0] = (__int64)v15;
  sub_2A37520(v14, "true", (__int64)"");
  v16 = 0;
  v17 = 0;
  v8 = sub_2A38130(a4, (__int64)v12);
  sub_B18290(v8, ".", 1u);
  if ( (_QWORD *)v14[0] != v15 )
    j_j___libc_free_0(v14[0]);
  if ( (_QWORD *)v12[0] != v13 )
    j_j___libc_free_0(v12[0]);
  if ( !a1 )
  {
    if ( a2 )
      return;
    goto LABEL_39;
  }
  if ( *a1 && a2 )
  {
    if ( *a1 )
      goto LABEL_7;
    goto LABEL_26;
  }
LABEL_6:
  sub_B17B50(a4);
  if ( *a1 )
    goto LABEL_7;
LABEL_26:
  sub_B18290(a4, " Inlined: ", 0xAu);
  v12[0] = (__int64)v13;
  sub_2A37520(v12, "StoreInlined", (__int64)"");
  v14[0] = (__int64)v15;
  sub_2A37520(v14, "false", (__int64)"");
  v16 = 0;
  v17 = 0;
  v9 = sub_2A38130(a4, (__int64)v12);
  sub_B18290(v9, ".", 1u);
  if ( (_QWORD *)v14[0] != v15 )
    j_j___libc_free_0(v14[0]);
  if ( (_QWORD *)v12[0] != v13 )
  {
    j_j___libc_free_0(v12[0]);
    if ( a2 )
      goto LABEL_8;
    goto LABEL_30;
  }
LABEL_7:
  if ( a2 )
    goto LABEL_8;
LABEL_30:
  sub_B18290(a4, " Volatile: ", 0xBu);
  v12[0] = (__int64)v13;
  sub_2A37520(v12, "StoreVolatile", (__int64)"");
  v14[0] = (__int64)v15;
  sub_2A37520(v14, "false", (__int64)"");
  v16 = 0;
  v17 = 0;
  v10 = sub_2A38130(a4, (__int64)v12);
  sub_B18290(v10, ".", 1u);
  if ( (_QWORD *)v14[0] != v15 )
    j_j___libc_free_0(v14[0]);
  if ( (_QWORD *)v12[0] != v13 )
  {
    j_j___libc_free_0(v12[0]);
    if ( a3 )
      return;
    goto LABEL_34;
  }
LABEL_8:
  if ( a3 )
    return;
LABEL_34:
  sub_B18290(a4, " Atomic: ", 9u);
  v12[0] = (__int64)v13;
  sub_2A37520(v12, "StoreAtomic", (__int64)"");
  v14[0] = (__int64)v15;
  sub_2A37520(v14, "false", (__int64)"");
  v16 = 0;
  v17 = 0;
  v11 = sub_2A38130(a4, (__int64)v12);
  sub_B18290(v11, ".", 1u);
  if ( (_QWORD *)v14[0] != v15 )
    j_j___libc_free_0(v14[0]);
  if ( (_QWORD *)v12[0] != v13 )
    j_j___libc_free_0(v12[0]);
}
