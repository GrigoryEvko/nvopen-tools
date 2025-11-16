// Function: sub_C2EF90
// Address: 0xc2ef90
//
__int64 __fastcall sub_C2EF90(__int64 a1, const void *a2, size_t a3)
{
  size_t v5; // rax
  __int64 result; // rax
  _BYTE *v7; // rdi
  _BYTE *v8; // [rsp+0h] [rbp-D0h] BYREF
  size_t v9; // [rsp+8h] [rbp-C8h]
  __int64 v10; // [rsp+10h] [rbp-C0h]
  _BYTE dest[184]; // [rsp+18h] [rbp-B8h] BYREF

  v8 = dest;
  v9 = 0;
  v10 = 128;
  if ( a3 > 0x80 )
  {
    sub_C8D290(&v8, dest, a3, 1);
    v7 = &v8[v9];
  }
  else
  {
    v5 = a3;
    if ( !a3 )
      goto LABEL_3;
    v7 = dest;
  }
  memcpy(v7, a2, a3);
  v5 = a3 + v9;
LABEL_3:
  v9 = v5;
  sub_C849A0(&v8);
  sub_CB6200(a1, v8, v9);
  result = sub_CB5D20(a1, 0);
  if ( v8 != dest )
    return _libc_free(v8, 0);
  return result;
}
