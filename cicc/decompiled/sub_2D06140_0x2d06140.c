// Function: sub_2D06140
// Address: 0x2d06140
//
unsigned __int64 __fastcall sub_2D06140(__int64 a1, const char *a2, const char **a3, _BYTE *a4, __int64 *a5)
{
  size_t v8; // rax
  const char *v9; // r13
  size_t v10; // rax
  size_t v11; // r9
  _QWORD *v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v16; // rax
  _QWORD *v17; // rdi
  size_t n; // [rsp+0h] [rbp-70h]
  size_t v20; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 v21[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v22[8]; // [rsp+30h] [rbp-40h] BYREF

  sub_D95050(a1, 0, 0);
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 136) = a1 + 152;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 168) = &unk_49DC130;
  *(_BYTE *)(a1 + 192) = 0;
  *(_BYTE *)(a1 + 208) = 0;
  *(_QWORD *)a1 = &unk_49DC010;
  *(_QWORD *)(a1 + 216) = &unk_49DC350;
  *(_QWORD *)(a1 + 248) = nullsub_92;
  *(_QWORD *)(a1 + 240) = sub_BC4D70;
  v8 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v8);
  v9 = *a3;
  v21[0] = (unsigned __int64)v22;
  if ( !v9 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v10 = strlen(v9);
  v20 = v10;
  v11 = v10;
  if ( v10 > 0xF )
  {
    n = v10;
    v16 = sub_22409D0((__int64)v21, &v20, 0);
    v11 = n;
    v21[0] = v16;
    v17 = (_QWORD *)v16;
    v22[0] = v20;
  }
  else
  {
    if ( v10 == 1 )
    {
      LOBYTE(v22[0]) = *v9;
      v12 = v22;
      goto LABEL_5;
    }
    if ( !v10 )
    {
      v12 = v22;
      goto LABEL_5;
    }
    v17 = v22;
  }
  memcpy(v17, v9, v11);
  v10 = v20;
  v12 = (_QWORD *)v21[0];
LABEL_5:
  v21[1] = v10;
  *((_BYTE *)v12 + v10) = 0;
  sub_2240AE0((unsigned __int64 *)(a1 + 136), v21);
  *(_BYTE *)(a1 + 208) = 1;
  sub_2240AE0((unsigned __int64 *)(a1 + 176), v21);
  if ( (_QWORD *)v21[0] != v22 )
    j_j___libc_free_0(v21[0]);
  v13 = *a5;
  *(_BYTE *)(a1 + 12) = (32 * (*a4 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v14 = a5[1];
  *(_QWORD *)(a1 + 40) = v13;
  *(_QWORD *)(a1 + 48) = v14;
  return sub_C53130(a1);
}
