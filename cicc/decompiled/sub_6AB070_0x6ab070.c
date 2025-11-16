// Function: sub_6AB070
// Address: 0x6ab070
//
__int64 __fastcall sub_6AB070(__int64 a1, __int64 a2)
{
  unsigned int *v3; // rsi
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdx
  _BYTE v9[368]; // [rsp+0h] [rbp-170h] BYREF

  sub_7AE210(a1);
  sub_7BC000(a1);
  sub_69ED20((__int64)v9, 0, 0, 1);
  sub_6F6C80(v9);
  v3 = 0;
  v4 = (__int64)v9;
  v5 = sub_6F6F40(v9, 0);
  v7 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(v5 + 16) = v7;
  *(_QWORD *)(a2 + 16) = v5;
  if ( word_4F06418[0] != 9 )
  {
    v3 = &dword_4F063F8;
    v4 = 18;
    sub_6851C0(0x12u, &dword_4F063F8);
    while ( word_4F06418[0] != 9 )
      sub_7B8B50(18, &dword_4F063F8, v7, v6);
  }
  return sub_7B8B50(v4, v3, v7, v6);
}
