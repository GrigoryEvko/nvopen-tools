// Function: sub_120D010
// Address: 0x120d010
//
__int64 __fastcall sub_120D010(__int64 a1, int a2, __int64 *a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  int v6; // eax
  unsigned __int64 v7; // rsi
  const char *v8; // rax
  int v9; // eax
  unsigned __int64 v10; // r13
  _QWORD v11[4]; // [rsp-58h] [rbp-58h] BYREF
  char v12; // [rsp-38h] [rbp-38h]
  char v13; // [rsp-37h] [rbp-37h]

  *a3 = 0;
  if ( a2 != *(_DWORD *)(a1 + 240) )
    return 0;
  v4 = a1 + 176;
  v6 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v6;
  if ( v6 != 12 )
  {
    v13 = 1;
    v7 = *(_QWORD *)(a1 + 232);
    v8 = "expected '('";
LABEL_5:
    v11[0] = v8;
    v12 = 3;
    sub_11FD800(v4, v7, (__int64)v11, 1);
    return 1;
  }
  v9 = sub_1205200(v4);
  v10 = *(_QWORD *)(a1 + 232);
  *(_DWORD *)(a1 + 240) = v9;
  result = sub_120C050(a1, a3);
  if ( !(_BYTE)result )
  {
    v7 = *(_QWORD *)(a1 + 232);
    if ( *(_DWORD *)(a1 + 240) != 13 )
    {
      v13 = 1;
      v8 = "expected ')'";
      goto LABEL_5;
    }
    *(_DWORD *)(a1 + 240) = sub_1205200(v4);
    result = 0;
    if ( !*a3 )
    {
      v13 = 1;
      v11[0] = "dereferenceable bytes must be non-zero";
      v12 = 3;
      sub_11FD800(v4, v10, (__int64)v11, 1);
      return 1;
    }
  }
  return result;
}
