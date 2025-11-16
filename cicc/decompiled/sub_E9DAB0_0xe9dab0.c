// Function: sub_E9DAB0
// Address: 0xe9dab0
//
__int64 __fastcall sub_E9DAB0(__int64 a1, const void *a2, signed __int64 a3, __int64 a4)
{
  __int64 (*v6)(void); // rdx
  __int64 v7; // rax
  char *v8; // rbx
  char *v9; // rax
  __int64 result; // rax
  __int64 v11; // [rsp+0h] [rbp-90h] BYREF
  int v12; // [rsp+8h] [rbp-88h]
  __int64 v13; // [rsp+10h] [rbp-80h]
  char v14; // [rsp+20h] [rbp-70h]
  __int64 v15; // [rsp+28h] [rbp-68h]
  char *v16; // [rsp+30h] [rbp-60h]
  char *v17; // [rsp+38h] [rbp-58h]
  char *v18; // [rsp+40h] [rbp-50h]
  _QWORD *v19; // [rsp+48h] [rbp-48h] BYREF
  _QWORD v20[7]; // [rsp+58h] [rbp-38h] BYREF

  v6 = *(__int64 (**)(void))(*(_QWORD *)a1 + 88LL);
  v7 = 1;
  if ( v6 != sub_E97650 )
    v7 = v6();
  v11 = v7;
  v14 = 10;
  v15 = a4;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  if ( a3 < 0 )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v8 = 0;
  if ( a3 )
  {
    v9 = (char *)sub_22077B0(a3);
    v8 = &v9[a3];
    v16 = v9;
    v18 = &v9[a3];
    memcpy(v9, a2, a3);
  }
  v17 = v8;
  v19 = v20;
  sub_E97AA0((__int64 *)&v19, byte_3F871B3, (__int64)byte_3F871B3);
  v13 = 0;
  v12 = 0;
  result = sub_E99320(a1);
  if ( result )
    result = sub_E9CD20((__int64 *)(result + 32), (__int64)&v11);
  if ( v19 != v20 )
    result = j_j___libc_free_0(v19, v20[0] + 1LL);
  if ( v16 )
    return j_j___libc_free_0(v16, v18 - v16);
  return result;
}
