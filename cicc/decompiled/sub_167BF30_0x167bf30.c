// Function: sub_167BF30
// Address: 0x167bf30
//
__int64 *__fastcall sub_167BF30(__int64 *a1, __int64 **a2, __int64 a3)
{
  __int64 *v5; // r13
  __int64 *(__fastcall *v6)(__int64 *, __int64 *); // rax
  _BYTE *v7; // r9
  size_t v8; // r8
  _QWORD *v9; // rax
  __int64 *v10; // rax
  __int64 v12; // rax
  _QWORD *v13; // rdi
  size_t n; // [rsp+0h] [rbp-E0h]
  _BYTE *src; // [rsp+8h] [rbp-D8h]
  __int64 v16; // [rsp+18h] [rbp-C8h]
  size_t v17; // [rsp+38h] [rbp-A8h] BYREF
  _QWORD v18[2]; // [rsp+40h] [rbp-A0h] BYREF
  _QWORD v19[2]; // [rsp+50h] [rbp-90h] BYREF
  _QWORD v20[2]; // [rsp+60h] [rbp-80h] BYREF
  _QWORD v21[2]; // [rsp+70h] [rbp-70h] BYREF
  void *v22; // [rsp+80h] [rbp-60h] BYREF
  __int64 v23; // [rsp+88h] [rbp-58h]
  __int64 v24; // [rsp+90h] [rbp-50h]
  __int64 v25; // [rsp+98h] [rbp-48h]
  int v26; // [rsp+A0h] [rbp-40h]
  _QWORD *v27; // [rsp+A8h] [rbp-38h]

  if ( !(*(unsigned __int8 (__fastcall **)(__int64 *, void *))(**a2 + 48))(*a2, &unk_4FA032B) )
  {
    v10 = *a2;
    *a2 = 0;
    *a1 = (unsigned __int64)v10 | 1;
    return a1;
  }
  v5 = *a2;
  *a2 = 0;
  v16 = **(_QWORD **)a3;
  v6 = *(__int64 *(__fastcall **)(__int64 *, __int64 *))(*v5 + 24);
  if ( v6 == sub_12BD5E0 )
  {
    LOBYTE(v21[0]) = 0;
    v20[0] = v21;
    v20[1] = 0;
    v26 = 1;
    v25 = 0;
    v24 = 0;
    v23 = 0;
    v22 = &unk_49EFBE0;
    v27 = v20;
    (*(void (__fastcall **)(__int64 *, void **))(*v5 + 16))(v5, &v22);
    if ( v25 != v23 )
      sub_16E7BA0(&v22);
    v18[0] = v19;
    v7 = (_BYTE *)*v27;
    v8 = v27[1];
    if ( v8 + *v27 && !v7 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v17 = v27[1];
    if ( v8 > 0xF )
    {
      n = v8;
      src = v7;
      v12 = sub_22409D0(v18, &v17, 0);
      v7 = src;
      v8 = n;
      v18[0] = v12;
      v13 = (_QWORD *)v12;
      v19[0] = v17;
    }
    else
    {
      if ( v8 == 1 )
      {
        LOBYTE(v19[0]) = *v7;
        v9 = v19;
        goto LABEL_10;
      }
      if ( !v8 )
      {
        v9 = v19;
        goto LABEL_10;
      }
      v13 = v19;
    }
    memcpy(v13, v7, v8);
    v8 = v17;
    v9 = (_QWORD *)v18[0];
LABEL_10:
    v18[1] = v8;
    *((_BYTE *)v9 + v8) = 0;
    sub_16E7BC0(&v22);
    if ( (_QWORD *)v20[0] != v21 )
      j_j___libc_free_0(v20[0], v21[0] + 1LL);
    goto LABEL_12;
  }
  v6(v18, v5);
LABEL_12:
  LOWORD(v21[0]) = 260;
  v20[0] = v18;
  sub_1670450((__int64)&v22, 0, (__int64)v20);
  sub_16027F0(v16, (__int64)&v22);
  if ( (_QWORD *)v18[0] != v19 )
    j_j___libc_free_0(v18[0], v19[0] + 1LL);
  **(_BYTE **)(a3 + 8) = 1;
  *a1 = 1;
  (*(void (__fastcall **)(__int64 *))(*v5 + 8))(v5);
  return a1;
}
