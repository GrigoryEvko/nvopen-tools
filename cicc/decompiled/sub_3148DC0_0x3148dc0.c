// Function: sub_3148DC0
// Address: 0x3148dc0
//
void __fastcall sub_3148DC0(__int64 a1, const char *a2, __int64 *a3, const char **a4)
{
  size_t v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  const char *v10; // r14
  size_t v11; // rax
  size_t v12; // r15
  _QWORD *v13; // rdx
  _QWORD *v14; // rdi
  size_t v15; // [rsp+8h] [rbp-58h] BYREF
  unsigned __int64 v16[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v17[8]; // [rsp+20h] [rbp-40h] BYREF

  v7 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v7);
  v8 = *a3;
  v9 = a3[1];
  v16[0] = (unsigned __int64)v17;
  *(_QWORD *)(a1 + 40) = v8;
  *(_QWORD *)(a1 + 48) = v9;
  v10 = *a4;
  if ( !v10 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v11 = strlen(v10);
  v15 = v11;
  v12 = v11;
  if ( v11 > 0xF )
  {
    v16[0] = sub_22409D0((__int64)v16, &v15, 0);
    v14 = (_QWORD *)v16[0];
    v17[0] = v15;
  }
  else
  {
    if ( v11 == 1 )
    {
      LOBYTE(v17[0]) = *v10;
      v13 = v17;
      goto LABEL_5;
    }
    if ( !v11 )
    {
      v13 = v17;
      goto LABEL_5;
    }
    v14 = v17;
  }
  memcpy(v14, v10, v12);
  v11 = v15;
  v13 = (_QWORD *)v16[0];
LABEL_5:
  v16[1] = v11;
  *((_BYTE *)v13 + v11) = 0;
  sub_2240AE0((unsigned __int64 *)(a1 + 136), v16);
  *(_BYTE *)(a1 + 208) = 1;
  sub_2240AE0((unsigned __int64 *)(a1 + 176), v16);
  if ( (_QWORD *)v16[0] != v17 )
    j_j___libc_free_0(v16[0]);
}
