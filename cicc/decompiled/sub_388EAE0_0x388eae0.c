// Function: sub_388EAE0
// Address: 0x388eae0
//
__int64 __fastcall sub_388EAE0(__int64 *a1, _QWORD *a2)
{
  char v2; // r8
  __int64 result; // rax
  _QWORD *v4; // rax
  _QWORD *v5; // r13
  const char *v6; // rax
  unsigned __int64 v7; // rsi
  char v8; // [rsp+Bh] [rbp-45h] BYREF
  int v9; // [rsp+Ch] [rbp-44h] BYREF
  const char *v10; // [rsp+10h] [rbp-40h] BYREF
  char v11; // [rsp+20h] [rbp-30h]
  char v12; // [rsp+21h] [rbp-2Fh]

  v9 = 0;
  v8 = 1;
  v2 = sub_388CFF0((__int64)a1, 1u, &v8, &v9);
  result = 1;
  if ( v2 )
    return result;
  if ( v9 == 1 )
  {
    v12 = 1;
    v6 = "fence cannot be unordered";
    goto LABEL_9;
  }
  if ( v9 == 2 )
  {
    v12 = 1;
    v6 = "fence cannot be monotonic";
LABEL_9:
    v7 = a1[7];
    v10 = v6;
    v11 = 3;
    return (unsigned __int8)sub_38814C0((__int64)(a1 + 1), v7, (__int64)&v10);
  }
  v4 = sub_1648A60(64, 0);
  v5 = v4;
  if ( v4 )
    sub_15F9C80((__int64)v4, *a1, v9, v8, 0);
  *a2 = v5;
  return 0;
}
