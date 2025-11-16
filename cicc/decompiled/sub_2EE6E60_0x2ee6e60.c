// Function: sub_2EE6E60
// Address: 0x2ee6e60
//
unsigned __int64 __fastcall sub_2EE6E60(unsigned int *a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rdx
  unsigned __int64 result; // rax
  __int64 v8; // r13
  _WORD *v9; // rdx
  __int64 v10; // [rsp-10h] [rbp-50h]
  _BYTE v11[16]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v12)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-30h]
  void (__fastcall *v13)(_BYTE *, __int64); // [rsp+18h] [rbp-28h]

  v3 = *(_QWORD *)a1;
  v4 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(**(_QWORD **)a1 + 16LL) + 200LL))(*(_QWORD *)(**(_QWORD **)a1 + 16LL));
  v5 = a1[2];
  sub_2FF6320(v11, v5, v4, 0, v3);
  if ( !v12 )
    sub_4263D6(v11, v5, v6);
  v13(v11, a2);
  result = (unsigned __int64)v12;
  if ( v12 )
    result = v12(v11, v11, 3);
  if ( a1[2] )
  {
    result = sub_2EBEE90(*(_QWORD *)a1, a1[2]);
    v8 = result;
    if ( result )
    {
      v9 = *(_WORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v9 <= 1u )
      {
        sub_CB6200(a2, (unsigned __int8 *)": ", 2u);
      }
      else
      {
        *v9 = 8250;
        *(_QWORD *)(a2 + 32) += 2LL;
      }
      sub_2E91850(v8, a2, 1u, 0, 0, 1, 0);
      return v10;
    }
  }
  return result;
}
