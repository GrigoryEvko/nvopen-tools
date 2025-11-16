// Function: sub_388D030
// Address: 0x388d030
//
__int64 __fastcall sub_388D030(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // r14
  int v5; // eax
  unsigned __int64 v6; // rsi
  const char *v7; // rax
  int v8; // eax
  unsigned __int64 v9; // r13
  _QWORD v10[2]; // [rsp-48h] [rbp-48h] BYREF
  char v11; // [rsp-38h] [rbp-38h]
  char v12; // [rsp-37h] [rbp-37h]

  *a2 = 0;
  if ( *(_DWORD *)(a1 + 64) != 96 )
    return 0;
  v3 = a1 + 8;
  v5 = sub_3887100(a1 + 8);
  *(_DWORD *)(a1 + 64) = v5;
  if ( v5 != 12 )
  {
    v12 = 1;
    v6 = *(_QWORD *)(a1 + 56);
    v7 = "expected '('";
LABEL_5:
    v10[0] = v7;
    v11 = 3;
    return sub_38814C0(v3, v6, (__int64)v10);
  }
  v8 = sub_3887100(v3);
  v9 = *(_QWORD *)(a1 + 56);
  *(_DWORD *)(a1 + 64) = v8;
  result = sub_388BA90(a1, a2);
  if ( !(_BYTE)result )
  {
    v6 = *(_QWORD *)(a1 + 56);
    if ( *(_DWORD *)(a1 + 64) != 13 )
    {
      v12 = 1;
      v7 = "expected ')'";
      goto LABEL_5;
    }
    *(_DWORD *)(a1 + 64) = sub_3887100(v3);
    if ( *a2 && (*a2 & (*a2 - 1)) == 0 )
    {
      return 0;
    }
    else
    {
      v12 = 1;
      v11 = 3;
      v10[0] = "stack alignment is not a power of two";
      return sub_38814C0(v3, v9, (__int64)v10);
    }
  }
  return result;
}
