// Function: sub_17C5B30
// Address: 0x17c5b30
//
__int64 __fastcall sub_17C5B30(__int64 a1, const char *a2, __int64 *a3, const char **a4)
{
  size_t v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  const char *v10; // r14
  size_t v11; // rax
  size_t v12; // r15
  _QWORD *v13; // rdx
  __int64 result; // rax
  _QWORD *v15; // rdi
  size_t v16; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v17[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v18[8]; // [rsp+20h] [rbp-40h] BYREF

  v7 = strlen(a2);
  sub_16B8280(a1, a2, v7);
  v8 = *a3;
  v9 = a3[1];
  v17[0] = v18;
  *(_QWORD *)(a1 + 40) = v8;
  *(_QWORD *)(a1 + 48) = v9;
  v10 = *a4;
  if ( !v10 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v11 = strlen(v10);
  v16 = v11;
  v12 = v11;
  if ( v11 > 0xF )
  {
    v17[0] = sub_22409D0(v17, &v16, 0);
    v15 = (_QWORD *)v17[0];
    v18[0] = v16;
  }
  else
  {
    if ( v11 == 1 )
    {
      LOBYTE(v18[0]) = *v10;
      v13 = v18;
      goto LABEL_5;
    }
    if ( !v11 )
    {
      v13 = v18;
      goto LABEL_5;
    }
    v15 = v18;
  }
  memcpy(v15, v10, v12);
  v11 = v16;
  v13 = (_QWORD *)v17[0];
LABEL_5:
  v17[1] = v11;
  *((_BYTE *)v13 + v11) = 0;
  sub_2240AE0(a1 + 160, v17);
  *(_BYTE *)(a1 + 232) = 1;
  result = sub_2240AE0(a1 + 200, v17);
  if ( (_QWORD *)v17[0] != v18 )
    return j_j___libc_free_0(v17[0], v18[0] + 1LL);
  return result;
}
