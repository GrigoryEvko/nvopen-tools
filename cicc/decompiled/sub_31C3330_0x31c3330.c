// Function: sub_31C3330
// Address: 0x31c3330
//
__int64 __fastcall sub_31C3330(__int64 a1, const char *a2, const char **a3, _BYTE *a4, _QWORD *a5)
{
  size_t v8; // rax
  const char *v9; // r13
  size_t v10; // rax
  size_t v11; // r9
  _QWORD *v12; // rdx
  __int64 result; // rax
  __int64 v14; // rax
  _QWORD *v15; // rdi
  size_t n; // [rsp+0h] [rbp-70h]
  size_t v18; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 v19[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v20[8]; // [rsp+30h] [rbp-40h] BYREF

  v8 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v8);
  v9 = *a3;
  v19[0] = (unsigned __int64)v20;
  if ( !v9 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v10 = strlen(v9);
  v18 = v10;
  v11 = v10;
  if ( v10 > 0xF )
  {
    n = v10;
    v14 = sub_22409D0((__int64)v19, &v18, 0);
    v11 = n;
    v19[0] = v14;
    v15 = (_QWORD *)v14;
    v20[0] = v18;
  }
  else
  {
    if ( v10 == 1 )
    {
      LOBYTE(v20[0]) = *v9;
      v12 = v20;
      goto LABEL_5;
    }
    if ( !v10 )
    {
      v12 = v20;
      goto LABEL_5;
    }
    v15 = v20;
  }
  memcpy(v15, v9, v11);
  v10 = v18;
  v12 = (_QWORD *)v19[0];
LABEL_5:
  v19[1] = v10;
  *((_BYTE *)v12 + v10) = 0;
  sub_2240AE0((unsigned __int64 *)(a1 + 136), v19);
  *(_BYTE *)(a1 + 208) = 1;
  sub_2240AE0((unsigned __int64 *)(a1 + 176), v19);
  if ( (_QWORD *)v19[0] != v20 )
    j_j___libc_free_0(v19[0]);
  *(_BYTE *)(a1 + 12) = (32 * (*a4 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  result = a5[1];
  *(_QWORD *)(a1 + 40) = *a5;
  *(_QWORD *)(a1 + 48) = result;
  return result;
}
