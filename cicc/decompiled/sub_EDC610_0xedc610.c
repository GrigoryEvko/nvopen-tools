// Function: sub_EDC610
// Address: 0xedc610
//
__int64 __fastcall sub_EDC610(__int64 a1, void **a2, __int64 *a3, void **a4)
{
  char v7; // al
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  __int64 *v11; // rsi
  void *v12; // rbx
  __int64 v13; // rax
  void *v14; // rdi
  char v15; // al
  __int64 v16; // rax
  unsigned __int64 v17; // r13
  char v18; // al
  bool v19; // bl
  __int64 v20; // [rsp+8h] [rbp-A8h] BYREF
  __int64 v21; // [rsp+10h] [rbp-A0h] BYREF
  char v22; // [rsp+18h] [rbp-98h]
  void *v23; // [rsp+20h] [rbp-90h] BYREF
  char v24; // [rsp+28h] [rbp-88h]
  __int64 v25[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v26; // [rsp+40h] [rbp-70h] BYREF
  void *v27[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v28; // [rsp+70h] [rbp-40h]

  sub_ED7EC0((__int64)&v21, a2, a3);
  v7 = v22;
  v22 &= ~2u;
  if ( (v7 & 1) != 0 )
  {
    v8 = v21;
    v21 = 0;
    v9 = v8 & 0xFFFFFFFFFFFFFFFELL;
    if ( v9 )
    {
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v9;
      return a1;
    }
  }
  v11 = (__int64 *)a4;
  sub_CA0F50(v25, a4);
  if ( !v25[1] )
  {
    v12 = 0;
    goto LABEL_7;
  }
  v27[0] = v25;
  v11 = (__int64 *)v27;
  v28 = 260;
  sub_ED7EC0((__int64)&v23, v27, a3);
  v15 = v24;
  v24 &= ~2u;
  if ( (v15 & 1) == 0 )
  {
    v27[0] = 0;
    sub_9C66B0((__int64 *)v27);
    v20 = 0;
    sub_9C66B0(&v20);
    goto LABEL_27;
  }
  v16 = (__int64)v23;
  v23 = 0;
  v27[0] = 0;
  v20 = v16 | 1;
  v17 = v16 & 0xFFFFFFFFFFFFFFFELL;
  sub_9C8CB0((__int64 *)v27);
  v18 = v24;
  v19 = (v24 & 2) != 0;
  if ( !v17 )
  {
    v20 = 0;
    sub_9C66B0(&v20);
    if ( v19 )
LABEL_25:
      sub_EDC5A0(&v23, (__int64)v27);
LABEL_27:
    v12 = v23;
    v23 = 0;
    if ( (v24 & 1) != 0 )
      sub_9C8CB0((__int64 *)&v23);
LABEL_7:
    v27[0] = v12;
    if ( (v22 & 2) != 0 )
      goto LABEL_23;
    v13 = v21;
    v11 = (__int64 *)&v23;
    v21 = 0;
    v23 = (void *)v13;
    sub_ED8AA0(a1, (__int64 *)&v23, (__int64 *)v27);
    if ( v23 )
      (*(void (__fastcall **)(void *))(*(_QWORD *)v23 + 8LL))(v23);
    v14 = v27[0];
    if ( !v27[0] )
      goto LABEL_12;
    goto LABEL_11;
  }
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v17;
  if ( v19 )
    goto LABEL_25;
  if ( (v18 & 1) != 0 )
  {
    sub_9C8CB0((__int64 *)&v23);
  }
  else
  {
    v14 = v23;
    if ( v23 )
LABEL_11:
      (*(void (__fastcall **)(void *))(*(_QWORD *)v14 + 8LL))(v14);
  }
LABEL_12:
  if ( (__int64 *)v25[0] != &v26 )
  {
    v11 = (__int64 *)(v26 + 1);
    j_j___libc_free_0(v25[0], v26 + 1);
  }
  if ( (v22 & 2) != 0 )
LABEL_23:
    sub_EDC5A0(&v21, (__int64)v11);
  if ( v21 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
  return a1;
}
