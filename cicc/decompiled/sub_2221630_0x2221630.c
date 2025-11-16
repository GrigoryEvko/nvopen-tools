// Function: sub_2221630
// Address: 0x2221630
//
void __fastcall sub_2221630(__int64 a1, __int64 *a2)
{
  void (__fastcall *v2)(__int64 *); // rax
  _BYTE *v3; // rbp
  unsigned __int64 v4; // r12
  void *v5; // rdi
  char *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // [rsp+8h] [rbp-40h] BYREF
  void *src[2]; // [rsp+10h] [rbp-38h] BYREF
  char v10; // [rsp+20h] [rbp-28h] BYREF

  (*(void (__fastcall **)(void **, __int64))(*(_QWORD *)a1 + 24LL))(src, a1);
  v2 = (void (__fastcall *)(__int64 *))a2[4];
  if ( v2 )
    v2(a2);
  v3 = src[0];
  v4 = (unsigned __int64)src[1];
  v5 = a2 + 2;
  *a2 = (__int64)(a2 + 2);
  if ( &v3[v4] && !v3 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v8 = v4;
  if ( v4 > 0xF )
  {
    v7 = sub_22409D0(a2, &v8, 0);
    *a2 = v7;
    v5 = (void *)v7;
    a2[2] = v8;
LABEL_14:
    memcpy(v5, v3, v4);
    v4 = v8;
    v5 = (void *)*a2;
    goto LABEL_8;
  }
  if ( v4 == 1 )
  {
    *((_BYTE *)a2 + 16) = *v3;
    goto LABEL_8;
  }
  if ( v4 )
    goto LABEL_14;
LABEL_8:
  a2[1] = v4;
  *((_BYTE *)v5 + v4) = 0;
  v6 = (char *)src[0];
  a2[4] = (__int64)sub_221F8D0;
  if ( v6 != &v10 )
    j___libc_free_0((unsigned __int64)v6);
}
