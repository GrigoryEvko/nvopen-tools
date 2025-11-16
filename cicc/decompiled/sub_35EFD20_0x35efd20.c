// Function: sub_35EFD20
// Address: 0x35efd20
//
_BYTE *__fastcall sub_35EFD20(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdi
  _BYTE *result; // rax
  _QWORD v11[6]; // [rsp+0h] [rbp-30h] BYREF

  sub_E82920(v11, a1, *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8));
  v9 = sub_CB6620(a4, (__int64)v11, v5, v6, v7, v8);
  result = *(_BYTE **)(v9 + 32);
  if ( *(_BYTE **)(v9 + 24) == result )
    return (_BYTE *)sub_CB6200(v9, (unsigned __int8 *)"U", 1u);
  *result = 85;
  ++*(_QWORD *)(v9 + 32);
  return result;
}
