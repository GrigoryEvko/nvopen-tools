// Function: sub_388C650
// Address: 0x388c650
//
__int64 __fastcall sub_388C650(__int64 a1, int a2, __int64 *a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  int v6; // eax
  unsigned __int64 v7; // rsi
  const char *v8; // rax
  int v9; // eax
  unsigned __int64 v10; // r13
  _QWORD v11[2]; // [rsp-48h] [rbp-48h] BYREF
  char v12; // [rsp-38h] [rbp-38h]
  char v13; // [rsp-37h] [rbp-37h]

  *a3 = 0;
  if ( a2 != *(_DWORD *)(a1 + 64) )
    return 0;
  v4 = a1 + 8;
  v6 = sub_3887100(a1 + 8);
  *(_DWORD *)(a1 + 64) = v6;
  if ( v6 != 12 )
  {
    v13 = 1;
    v7 = *(_QWORD *)(a1 + 56);
    v8 = "expected '('";
LABEL_5:
    v11[0] = v8;
    v12 = 3;
    return sub_38814C0(v4, v7, (__int64)v11);
  }
  v9 = sub_3887100(v4);
  v10 = *(_QWORD *)(a1 + 56);
  *(_DWORD *)(a1 + 64) = v9;
  result = sub_388BD80(a1, a3);
  if ( !(_BYTE)result )
  {
    v7 = *(_QWORD *)(a1 + 56);
    if ( *(_DWORD *)(a1 + 64) != 13 )
    {
      v13 = 1;
      v8 = "expected ')'";
      goto LABEL_5;
    }
    *(_DWORD *)(a1 + 64) = sub_3887100(v4);
    result = 0;
    if ( !*a3 )
    {
      v13 = 1;
      v12 = 3;
      v11[0] = "dereferenceable bytes must be non-zero";
      return sub_38814C0(v4, v10, (__int64)v11);
    }
  }
  return result;
}
