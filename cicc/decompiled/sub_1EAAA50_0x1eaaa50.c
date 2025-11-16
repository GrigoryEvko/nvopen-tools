// Function: sub_1EAAA50
// Address: 0x1eaaa50
//
__int64 __fastcall sub_1EAAA50(__int64 a1, const char *a2, __int64 *a3, const char **a4, _DWORD *a5)
{
  size_t v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  const char *v12; // r15
  size_t v13; // rax
  size_t v14; // r8
  _QWORD *v15; // rdx
  __int64 result; // rax
  __int64 v17; // rax
  _QWORD *v18; // rdi
  size_t n; // [rsp+8h] [rbp-68h]
  size_t v20; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v21[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v22[8]; // [rsp+30h] [rbp-40h] BYREF

  v9 = strlen(a2);
  sub_16B8280(a1, a2, v9);
  v10 = *a3;
  v11 = a3[1];
  v21[0] = v22;
  *(_QWORD *)(a1 + 40) = v10;
  *(_QWORD *)(a1 + 48) = v11;
  v12 = *a4;
  if ( !v12 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v13 = strlen(v12);
  v20 = v13;
  v14 = v13;
  if ( v13 > 0xF )
  {
    n = v13;
    v17 = sub_22409D0(v21, &v20, 0);
    v14 = n;
    v21[0] = v17;
    v18 = (_QWORD *)v17;
    v22[0] = v20;
  }
  else
  {
    if ( v13 == 1 )
    {
      LOBYTE(v22[0]) = *v12;
      v15 = v22;
      goto LABEL_5;
    }
    if ( !v13 )
    {
      v15 = v22;
      goto LABEL_5;
    }
    v18 = v22;
  }
  memcpy(v18, v12, v14);
  v13 = v20;
  v15 = (_QWORD *)v21[0];
LABEL_5:
  v21[1] = v13;
  *((_BYTE *)v15 + v13) = 0;
  sub_2240AE0(a1 + 160, v21);
  *(_BYTE *)(a1 + 232) = 1;
  sub_2240AE0(a1 + 200, v21);
  if ( (_QWORD *)v21[0] != v22 )
    j_j___libc_free_0(v21[0], v22[0] + 1LL);
  result = (32 * (*a5 & 3)) | *(_BYTE *)(a1 + 12) & 0x9Fu;
  *(_BYTE *)(a1 + 12) = (32 * (*(_BYTE *)a5 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  return result;
}
