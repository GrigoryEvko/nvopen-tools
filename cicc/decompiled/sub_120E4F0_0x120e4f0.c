// Function: sub_120E4F0
// Address: 0x120e4f0
//
__int64 __fastcall sub_120E4F0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // r14
  int v5; // eax
  unsigned __int64 v6; // rsi
  const char *v7; // rax
  int v8; // eax
  unsigned __int64 v9; // r13
  _QWORD v10[4]; // [rsp-58h] [rbp-58h] BYREF
  char v11; // [rsp-38h] [rbp-38h]
  char v12; // [rsp-37h] [rbp-37h]

  *a2 = 0;
  if ( *(_DWORD *)(a1 + 240) != 259 )
    return 0;
  v3 = a1 + 176;
  v5 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v5;
  if ( v5 != 12 )
  {
    v12 = 1;
    v6 = *(_QWORD *)(a1 + 232);
    v7 = "expected '('";
LABEL_5:
    v10[0] = v7;
    v11 = 3;
    sub_11FD800(v3, v6, (__int64)v10, 1);
    return 1;
  }
  v8 = sub_1205200(v3);
  v9 = *(_QWORD *)(a1 + 232);
  *(_DWORD *)(a1 + 240) = v8;
  result = sub_120BD00(a1, a2);
  if ( !(_BYTE)result )
  {
    v6 = *(_QWORD *)(a1 + 232);
    if ( *(_DWORD *)(a1 + 240) != 13 )
    {
      v12 = 1;
      v7 = "expected ')'";
      goto LABEL_5;
    }
    *(_DWORD *)(a1 + 240) = sub_1205200(v3);
    if ( *a2 && (*a2 & (*a2 - 1)) == 0 )
    {
      return 0;
    }
    else
    {
      v12 = 1;
      v10[0] = "stack alignment is not a power of two";
      v11 = 3;
      sub_11FD800(v3, v9, (__int64)v10, 1);
      return 1;
    }
  }
  return result;
}
