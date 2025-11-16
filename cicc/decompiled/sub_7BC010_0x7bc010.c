// Function: sub_7BC010
// Address: 0x7bc010
//
__int64 sub_7BC010()
{
  __int64 v0; // rsi
  __int64 v1; // rcx
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 v5; // rax
  _BYTE v6[48]; // [rsp+0h] [rbp-30h] BYREF

  sub_7ADF70((__int64)v6, 0);
  word_4F06418[0] = 44;
  v0 = *((unsigned int *)qword_4F061C0 + 2);
  if ( (_DWORD)v0 )
  {
    v5 = qword_4F061C0[5];
    if ( *(_DWORD *)(v5 + 28) == dword_4F06650[0] )
      *(_WORD *)(v5 + 24) = 44;
  }
  sub_7AE360((__int64)v6);
  dword_4F0664C = ++dword_4F06650[0];
  return sub_7BC000((unsigned __int64)v6, v0, (__int64)&dword_4F0664C, v1, v2, v3);
}
