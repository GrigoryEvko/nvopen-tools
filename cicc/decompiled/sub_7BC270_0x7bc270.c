// Function: sub_7BC270
// Address: 0x7bc270
//
__int64 __fastcall sub_7BC270(_BYTE *a1)
{
  int v1; // r15d
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 result; // rax
  _BYTE v15[80]; // [rsp+0h] [rbp-50h] BYREF

  v1 = dword_4F04D80;
  dword_4F04D80 = 1;
  sub_7ADF70((__int64)v15, 0);
  sub_7BC160((__int64)a1);
  while ( word_4F06418[0] != 9 )
  {
    a1 = v15;
    sub_7AE360((__int64)v15);
    sub_7B8B50((unsigned __int64)v15, 0, v6, v7, v8, v9);
  }
  sub_7B8B50((unsigned __int64)a1, 0, v2, v3, v4, v5);
  result = sub_7BC000((unsigned __int64)v15, 0, v10, v11, v12, v13);
  dword_4F04D80 = v1;
  return result;
}
