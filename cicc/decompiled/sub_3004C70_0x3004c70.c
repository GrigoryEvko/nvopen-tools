// Function: sub_3004C70
// Address: 0x3004c70
//
unsigned __int64 __fastcall sub_3004C70(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  _QWORD *v4; // rsi
  unsigned __int64 result; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  void *v8; // [rsp+8h] [rbp-18h] BYREF

  v3 = *(_QWORD **)(a2 + 112);
  v4 = &v3[*(unsigned int *)(a2 + 120)];
  v8 = &unk_4F8144C;
  result = (unsigned __int64)sub_3004BB0(v3, (__int64)v4, (__int64 *)&v8);
  if ( v4 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 124);
    if ( v6 + 1 > result )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, v7);
      result = *(_QWORD *)(a2 + 112);
      v4 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F8144C;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}
