// Function: sub_2222180
// Address: 0x2222180
//
void __fastcall sub_2222180(
        __int64 a1,
        __int64 *a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        _BYTE *a6,
        __int64 a7)
{
  void (__fastcall *v10)(__int64 *); // rax
  _BYTE *v11; // r13
  unsigned __int64 v12; // r12
  void *v13; // rdi
  char *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // [rsp+18h] [rbp-80h] BYREF
  __int64 v17[2]; // [rsp+20h] [rbp-78h] BYREF
  _BYTE v18[16]; // [rsp+30h] [rbp-68h] BYREF
  void *src[2]; // [rsp+40h] [rbp-58h] BYREF
  char v20; // [rsp+50h] [rbp-48h] BYREF

  v17[0] = (__int64)v18;
  sub_221FC40(v17, a6, (__int64)&a6[a7]);
  (*(void (__fastcall **)(void **, __int64, _QWORD, _QWORD, _QWORD, __int64 *))(*(_QWORD *)a1 + 24LL))(
    src,
    a1,
    a3,
    a4,
    a5,
    v17);
  v10 = (void (__fastcall *)(__int64 *))a2[4];
  if ( v10 )
    v10(a2);
  v11 = src[0];
  v12 = (unsigned __int64)src[1];
  v13 = a2 + 2;
  *a2 = (__int64)(a2 + 2);
  if ( &v11[v12] && !v11 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v16 = v12;
  if ( v12 > 0xF )
  {
    v15 = sub_22409D0(a2, &v16, 0);
    *a2 = v15;
    v13 = (void *)v15;
    a2[2] = v16;
LABEL_16:
    memcpy(v13, v11, v12);
    v12 = v16;
    v13 = (void *)*a2;
    goto LABEL_8;
  }
  if ( v12 == 1 )
  {
    *((_BYTE *)a2 + 16) = *v11;
    goto LABEL_8;
  }
  if ( v12 )
    goto LABEL_16;
LABEL_8:
  a2[1] = v12;
  *((_BYTE *)v13 + v12) = 0;
  v14 = (char *)src[0];
  a2[4] = (__int64)sub_221F8D0;
  if ( v14 != &v20 )
    j___libc_free_0((unsigned __int64)v14);
  if ( (_BYTE *)v17[0] != v18 )
    j___libc_free_0(v17[0]);
}
