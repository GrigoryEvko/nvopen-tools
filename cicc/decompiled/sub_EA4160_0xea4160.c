// Function: sub_EA4160
// Address: 0xea4160
//
__int64 __fastcall sub_EA4160(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v2; // r13
  void (__fastcall *v3)(_QWORD *, _QWORD, __int64); // rbx
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  const char *v7; // [rsp-58h] [rbp-58h] BYREF
  char v8; // [rsp-38h] [rbp-38h]
  char v9; // [rsp-37h] [rbp-37h]

  result = *(unsigned __int8 *)(a1 + 869);
  if ( (_BYTE)result )
    return 0;
  v2 = *(_QWORD **)(a1 + 232);
  if ( !v2[36] )
  {
    v3 = *(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*v2 + 192LL);
    v4 = sub_ECE6C0(*(_QWORD *)(a1 + 8));
    v3(v2, 0, v4);
    v9 = 1;
    v7 = "expected section directive before assembly directive";
    v8 = 3;
    v5 = sub_ECD7B0(a1);
    v6 = sub_ECD6A0(v5);
    return sub_ECDA70(a1, v6, &v7, 0, 0);
  }
  return result;
}
